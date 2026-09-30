"""Live viewer engine: one GPU thread runs the live clock and renders every connected viewer.

Each tick (1/fps): advance the replayed "live" feeds, sync + health-check them, run the online
method (visual hull) once, then render the latest state from each viewer's own camera and
JPEG-encode it. Viewers only send camera poses; the heavy work is shared.
"""
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from fvv.data import Camera, MultiViewSequence
from fvv.eval import StageTimer
from fvv.live import FaultConfig, ReplayStreams, Synchronizer
from fvv.render import look_at, scene_frame

DEMO_FAULTS = "jitter=40,drop=00_05@1-3,freeze=00_07@2-3,blur=00_03@0.5-4,corrupt=00_10@1.5,res=00_04@2x0.5"


@dataclass
class Viewer:
    """Orbit camera around the scene target: azimuth/elevation (deg), distance (cm), zoom (x)."""
    az: float = 30.0
    el: float = 25.0
    dist: float = 250.0
    zoom: float = 1.0
    quality: int = 80
    send: object = None                   # callable(bytes header_json, bytes jpeg), set by the web layer
    last_stats: dict = field(default_factory=dict)


class Engine:
    def __init__(self, seq: MultiViewSequence, cams: list[str], methods: dict, fps: float = 30.0):
        self.seq, self.cams, self.methods, self.fps = seq, cams, methods, fps
        self.frames = seq.frame_ids
        self.mode = "live" if "live" in methods else next(iter(methods))
        self.playing, self.k, self.faults_on = True, 0, False
        self.viewers: dict[int, Viewer] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

        print(f"[engine] loading {len(self.frames)} frames x {len(cams)} cameras into GPU memory ...", flush=True)
        self.streams = ReplayStreams(seq, cams, self.frames, FaultConfig.parse(""), fps)
        self.sync = Synchronizer(cams, {c: (seq.camera(c).width, seq.camera(c).height) for c in cams}, fps)

        # orbit frame + limits
        self.target, self.down = scene_frame(seq)
        C = np.stack([seq.camera(c).center for c in seq.camera_names])
        rel = C - self.target
        h = rel @ self.down
        horiz = rel - np.outer(h, self.down)
        self.e1 = horiz[0] / np.linalg.norm(horiz[0])
        self.e2 = np.cross(self.down, self.e1)
        P = np.stack([horiz @ self.e1, horiz @ self.e2], 1)
        self.fp_center, self.fp_r = P.mean(0), np.linalg.norm(P - P.mean(0), axis=1).mean()
        self.template = seq.camera(cams[0])
        # best source pixel density at the target (focal / distance): caps useful zoom
        self.src_density = max(seq.camera(c).K[0, 0] / np.linalg.norm(seq.camera(c).center - self.target)
                               for c in cams)
        self.state, self.fs, self.last_process_ms = None, None, 0.0

    # ---- viewer camera with limits -------------------------------------------------------
    def max_dist(self, az_rad: float) -> float:
        """Distance from target to 90% of the rig footprint along this azimuth (stay inside)."""
        d = np.array([np.cos(az_rad), np.sin(az_rad)])
        b = d @ self.fp_center
        return 0.9 * (b + np.sqrt(max(b * b - self.fp_center @ self.fp_center + self.fp_r ** 2, 0.0)))

    def camera(self, v: Viewer) -> tuple[Camera, dict]:
        az, el = np.radians(v.az), np.radians(np.clip(v.el, -5, 80))
        dist = float(np.clip(v.dist, 80.0, self.max_dist(az)))
        horiz = np.cos(az) * self.e1 + np.sin(az) * self.e2
        eye = self.target + dist * (np.cos(el) * horiz - np.sin(el) * self.down)
        R, t = look_at(eye, self.target, self.down)
        # zoom cap: virtual pixel density at the target <= 1.8x the best real camera's
        f0 = self.template.K[0, 0]
        zoom = float(np.clip(v.zoom, 0.5, 1.8 * self.src_density * dist / f0))
        K = self.template.K.copy()
        K[0, 0] *= zoom
        K[1, 1] *= zoom
        cam = Camera("viewer", K, np.zeros(5), R, t, self.template.width, self.template.height)
        return cam, {"dist": round(dist), "zoom": round(zoom, 2), "zoom_limited": zoom < v.zoom - 1e-3,
                     "dist_limited": abs(dist - v.dist) > 1}

    # ---- controls (called from the web thread) ------------------------------------------
    def set_faults(self, on: bool) -> None:
        self.faults_on = on
        self.streams.faults = FaultConfig.parse(DEMO_FAULTS if on else "")

    def seek(self, k: int) -> None:
        self.k = int(k) % len(self.frames)
        self.state = None

    # ---- main loop --------------------------------------------------------------------
    def run(self) -> None:
        tick = 1.0 / self.fps
        next_t = time.perf_counter()
        while not self.stop.is_set():
            with self.lock:
                idle = not self.viewers
            if idle:                               # nobody watching: keep the GPU free
                time.sleep(0.2)
                next_t = time.perf_counter()
                continue
            timer = StageTimer()
            t0 = time.perf_counter()
            if self.playing or self.state is None or self.fs is None:
                if self.playing and self.state is not None:
                    self.k = (self.k + 1) % len(self.frames)
                with timer("sync_health"):
                    self.fs = self.sync.assemble(self.k, self.frames[self.k], self.streams.packets(self.k))
                if self.mode == "live":
                    self.state = self.methods["live"].process(self.fs.images, timer)
                else:
                    self.state = "offline"
                self.last_process_ms = (time.perf_counter() - t0) * 1000

            with self.lock:
                viewers = list(self.viewers.values())
            frame = self.frames[self.k]
            for v in viewers:
                t1 = time.perf_counter()
                cam, info = self.camera(v)
                if self.mode == "live":
                    img = self.methods["live"].render_view(self.state, cam, timer)
                else:
                    img = self.methods[self.mode].render(frame, cam, timer)
                render_ms = (time.perf_counter() - t1) * 1000
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, int(v.quality)])
                header = {
                    "type": "frame", "slot": self.k, "frame": frame, "n": len(self.frames),
                    "mode": self.mode, "playing": self.playing, "faults": self.faults_on,
                    "process_ms": round(self.last_process_ms, 1), "render_ms": round(render_ms, 1),
                    "cams_ok": len(self.fs.images), "cams_total": len(self.cams),
                    "status": [self.fs.status.get(c, "missing") for c in self.cams], **info,
                }
                if v.send:
                    v.send(header, jpg.tobytes())

            next_t += tick
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()        # fell behind: don't try to catch up (live)
