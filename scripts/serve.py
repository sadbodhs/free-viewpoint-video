"""Local live viewer: open http://<3090-ip>:8080 on any device on the same network.

    scripts/serve.sh data/panoptic/170221_haggling_b1            # live hull + offline layered
    scripts/serve.sh data/panoptic/170221_haggling_b1 --no-layered

Modes: "live" = replayed camera feeds -> visual hull every frame (real-time path);
"layered" = precomputed offline 3D video (best quality, needs its checkpoints).
"""
import argparse
import threading

from aiohttp import web

from fvv.data import MultiViewSequence
from fvv.eval import load_or_create_split
from fvv.methods import make_method
from fvv.serve import Engine, make_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--no-layered", action="store_true", help="skip the offline layered mode")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    test = load_or_create_split(seq)
    cams = [c for c in seq.camera_names if c not in test]
    ckpt = f"outputs/models/{seq.root.name}"

    methods = {"live": make_method("hull", ckpt_dir=ckpt)}
    methods["live"].fit(seq, cams, seq.frame_ids)
    if not args.no_layered:
        methods["layered"] = make_method("layered", ckpt_dir=ckpt)
        methods["layered"].fit(seq, cams, seq.frame_ids)

    engine = Engine(seq, cams, methods)
    threading.Thread(target=engine.run, daemon=True).start()
    print(f"[serve] http://0.0.0.0:{args.port}  (open http://<this-machine-ip>:{args.port})", flush=True)
    web.run_app(make_app(engine), host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
