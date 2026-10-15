"""Train only the static background splat of a sequence (people masked out), which is all the
live visual hull needs. The offline layered method reuses the same checkpoint.

    scripts/run.sh python scripts/triangulate_points.py data/panoptic/<seq>
    scripts/run.sh python scripts/train_background.py data/panoptic/<seq>
"""
import argparse
from pathlib import Path

from fvv.data import MultiViewSequence
from fvv.eval import load_or_create_split
from fvv.methods.layered_splat import LayeredSplat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--keyframes", type=int, default=12)
    ap.add_argument("--iters", type=int, default=30_000)
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    test = load_or_create_split(seq)
    ckpt = Path(f"outputs/models/{seq.root.name}") / f"layered_bg_{seq.root.name}.pt"
    m = LayeredSplat(bg_frames=args.keyframes, bg_iters=args.iters)
    m.train_cams = [c for c in seq.camera_names if c not in test]
    m.cam_index = {c: k for k, c in enumerate(m.train_cams)}
    print(f"held-out {test}; training background from {len(m.train_cams)} cameras")
    m._fit_background(seq, seq.frame_ids)
    m.bg.save(ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
