"""Held-out comparison video: rows = held-out cameras, columns = ground truth + each method.

Uses predictions saved by scripts/evaluate.py (outputs/eval/<seq>/<method>/pred/) and overlays
each method's per-frame PSNR. Only frames evaluated by every method are shown.

    scripts/run.sh python scripts/make_comparison_video.py data/panoptic/170221_haggling_b1 \
        --methods nearest_view 3dgs
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fvv.data import MultiViewSequence
from fvv.render import VideoWriter, even, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--scale", type=float, default=0.5, help="must match evaluate.py")
    ap.add_argument("--tile-width", type=int, default=480)
    ap.add_argument("--hold", type=int, default=90, help="video frames per shown frame when there are few")
    ap.add_argument("--out", default="outputs/videos")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    ev = Path(args.eval_dir) / seq.root.name
    results = {m: json.loads((ev / m / "results.json").read_text()) for m in args.methods}
    psnr = {m: {(r["cam"], r["frame"]): r["psnr"] for r in res["rows"]} for m, res in results.items()}
    test_cams = results[args.methods[0]]["test_cams"]
    keys = sorted(set.intersection(*(set(p) for p in psnr.values())), key=lambda k: (k[1], k[0]))
    frames = sorted({f for _, f in keys})
    assert frames, "no frame was evaluated by all methods"
    hold = args.hold if len(frames) < 10 else 1

    tw = args.tile_width
    th = round(tw * seq.camera(test_cams[0]).height / seq.camera(test_cams[0]).width)

    def tile(img, text):
        return label(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), text, scale=0.6)

    out = Path(args.out) / seq.root.name
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"comparison_{'_vs_'.join(args.methods)}.mp4"
    blank = np.zeros((th, tw, 3), np.uint8)
    grid_w, grid_h = even(np.zeros((th * len(test_cams), tw * (1 + len(args.methods))))).shape[::-1]
    with VideoWriter(str(video), grid_w, grid_h) as vw:
        for f in frames:
            rows = []
            for c in test_cams:
                if (c, f) not in psnr[args.methods[0]]:
                    rows.append(np.concatenate([blank] * (1 + len(args.methods)), 1))
                    continue
                cells = [tile(seq.image(c, f), f"ground truth  {c}")]
                for m in args.methods:
                    pred = cv2.cvtColor(cv2.imread(str(ev / m / "pred" / c / f"{f:08d}.jpg")), cv2.COLOR_BGR2RGB)
                    cells.append(tile(pred, f"{m}  {psnr[m][(c, f)]:.1f} dB"))
                rows.append(np.concatenate(cells, 1))
            grid = even(np.concatenate(rows, 0))
            for _ in range(hold):
                vw.write(grid)
    print(video)


if __name__ == "__main__":
    main()
