"""Triangulate a sparse point cloud for one frame from the TRAINING cameras (held-out ones excluded).

Writes <root>/points/<frame:08d>.ply (world units, RGB), used to initialize Gaussian splatting.

    scripts/run.sh python scripts/triangulate_points.py data/panoptic/170221_haggling_b1
"""
import argparse

from fvv.data import MultiViewSequence
from fvv.eval import load_or_create_split
from fvv.geometry import save_points_ply, triangulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--frame", type=int, help="default: first frame where every camera is valid")
    ap.add_argument("--scale", type=float, default=1.0, help="image scale for feature matching")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    frame = args.frame if args.frame is not None else next(
        f for f in seq.frame_ids if len(seq.valid_cameras(f)) == len(seq.camera_names))
    test = set(load_or_create_split(seq))
    cams = [c for c in seq.valid_cameras(frame) if c not in test]
    print(f"frame {frame}: triangulating from {len(cams)} training cameras")

    pts, cols = triangulate(seq, frame, cams)
    out = seq.root / "points" / f"{frame:08d}.ply"
    save_points_ply(out, pts, cols)
    print(f"{len(pts):,} points -> {out}")


if __name__ == "__main__":
    main()
