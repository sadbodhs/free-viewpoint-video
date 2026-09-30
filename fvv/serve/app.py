"""aiohttp web layer: serves the viewer page and one WebSocket per viewer.

Viewer -> server: JSON {"type": "pose", az, el, dist, zoom} or {"type": "control", ...}.
Server -> viewer: binary [4-byte header length][JSON header][JPEG], newest frame only
(a slow client skips frames instead of building up latency).
"""
import asyncio
import itertools
import json
import struct
from pathlib import Path

from aiohttp import WSMsgType, web

from .engine import Engine, Viewer

STATIC = Path(__file__).parent / "static"


def make_app(engine: Engine) -> web.Application:
    ids = itertools.count()

    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=1 << 20)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        latest: asyncio.Queue = asyncio.Queue(maxsize=1)

        def send(header: dict, jpg: bytes) -> None:          # called from the engine thread
            msg = json.dumps(header).encode()
            payload = struct.pack("<I", len(msg)) + msg + jpg

            def put():
                if latest.full():
                    latest.get_nowait()
                latest.put_nowait(payload)
            loop.call_soon_threadsafe(put)

        vid = next(ids)
        viewer = Viewer(send=send)
        with engine.lock:
            engine.viewers[vid] = viewer
        await ws.send_json({"type": "init", "frames": len(engine.frames), "fps": engine.fps,
                            "cams": engine.cams, "modes": list(engine.methods), "mode": engine.mode,
                            "viewer": {"az": viewer.az, "el": viewer.el, "dist": viewer.dist, "zoom": viewer.zoom}})

        async def sender():
            while not ws.closed:
                payload = await latest.get()
                try:
                    await ws.send_bytes(payload)
                except ConnectionResetError:
                    break

        task = asyncio.create_task(sender())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                if m.get("type") == "pose":
                    for k in ("az", "el", "dist", "zoom", "quality"):
                        if k in m:
                            setattr(viewer, k, float(m[k]))
                elif m.get("type") == "control":
                    if "playing" in m:
                        engine.playing = bool(m["playing"])
                    if "seek" in m:
                        engine.seek(int(m["seek"]))
                    if m.get("mode") in engine.methods:
                        engine.mode = m["mode"]
                    if "faults" in m:
                        engine.set_faults(bool(m["faults"]))
        finally:
            task.cancel()
            with engine.lock:
                engine.viewers.pop(vid, None)
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", STATIC)
    return app
