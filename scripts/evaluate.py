"""Evaluate a novel-view method on held-out cameras.

    scripts/run.sh python scripts/evaluate.py data/panoptic/170221_haggling_b1 --method nearest_view
"""
import argparse
import json

from fvv.data import MultiViewSequence
from fvv.eval import evaluate, pick_test_cameras
from fvv.methods import METHODS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--method", default="nearest_view", choices=list(METHODS))
    ap.add_argument("--test-cams", nargs="*", help="default: 3 spread-out cameras")
    ap.add_argument("--frame-step", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.5, help="eval resolution (0.5 -> 960x540)")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    test_cams = args.test_cams or pick_test_cameras(seq)
    frames = seq.frame_ids[::args.frame_step]
    print(f"{len(seq.camera_names)} cams, test={test_cams}, {len(frames)} frames")

    res = evaluate(METHODS[args.method](), seq, test_cams, frames, f"{args.out}/{seq.root.name}")
    print(json.dumps({k: res[k] for k in ("method", "mean", "per_camera", "latency")}, indent=1))


if __name__ == "__main__":
    main()
