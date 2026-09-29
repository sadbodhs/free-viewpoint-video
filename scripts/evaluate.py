"""Evaluate a novel-view method on held-out cameras.

    scripts/run.sh python scripts/evaluate.py data/panoptic/170221_haggling_b1 --method nearest_view
    scripts/run.sh python scripts/evaluate.py data/panoptic/170221_haggling_b1 --method 3dgs
"""
import argparse
import json

from fvv.data import MultiViewSequence
from fvv.eval import evaluate, load_or_create_split
from fvv.methods import METHODS, make_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--method", default="nearest_view", choices=METHODS)
    ap.add_argument("--test-cams", nargs="*", help="default: fixed split in <root>/split.json")
    ap.add_argument("--frame-step", type=int, default=15)
    ap.add_argument("--frame", type=int, help="frame for single-frame methods; default first complete frame")
    ap.add_argument("--scale", type=float, default=0.5, help="eval resolution (0.5 -> 960x540)")
    ap.add_argument("--iters", type=int, default=30_000, help="3dgs optimization steps")
    ap.add_argument("--retrain", action="store_true", help="ignore existing checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    test_cams = args.test_cams or load_or_create_split(seq)
    method = make_method(args.method, iters=args.iters, retrain=args.retrain, seed=args.seed,
                         ckpt_dir=f"outputs/models/{seq.root.name}")

    if getattr(method, "single_frame", False):
        frame = args.frame if args.frame is not None else next(
            f for f in seq.frame_ids if len(seq.valid_cameras(f)) == len(seq.camera_names))
        frames = [frame]
    else:
        frames = seq.frame_ids[::args.frame_step]
    print(f"{len(seq.camera_names)} cams, test={test_cams}, frames={frames[0]}..{frames[-1]} ({len(frames)})")

    res = evaluate(method, seq, test_cams, frames, f"{args.out}/{seq.root.name}")
    print(json.dumps({k: res[k] for k in ("method", "mean", "per_camera", "latency")}, indent=1))


if __name__ == "__main__":
    main()
