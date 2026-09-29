"""Sanity-check calibration + sync by projecting 3D skeletons into every camera.

If calibration is right, skeletons sit on the people in every view. If a camera's
video is out of sync, its skeletons lag/lead the person during fast motion.
Writes a contact sheet per frame to outputs/calib_check/ and prints per-camera stats.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fvv.data import COCO19_EDGES, MultiViewSequence

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)]


def draw_bodies(img, cam, bodies, conf_thresh=0.1):
    """Draw projected skeletons; returns fraction of confident joints landing inside the image."""
    inside, total = 0, 0
    for b in bodies:
        J = b["joints"]
        uv, depth = cam.project(J[:, :3])
        ok = (J[:, 3] > conf_thresh) & (depth > 0)
        color = COLORS[b["id"] % len(COLORS)]
        for i, j in COCO19_EDGES:
            if ok[i] and ok[j]:
                cv2.line(img, tuple(np.int32(uv[i])), tuple(np.int32(uv[j])), color, 4, cv2.LINE_AA)
        for p in uv[ok]:
            cv2.circle(img, tuple(np.int32(p)), 6, color, -1, cv2.LINE_AA)
        in_img = ok & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        inside += in_img.sum()
        total += ok.sum()
    return inside / max(total, 1)


def contact_sheet(tiles, names, cols=6, tile_w=480):
    tile_h = round(tile_w * tiles[0].shape[0] / tiles[0].shape[1])
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
    for k, (t, n) in enumerate(zip(tiles, names)):
        t = cv2.resize(t, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(t, n, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        r, c = divmod(k, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = t
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--frames", type=int, nargs="*", help="frame ids; default first/middle/last")
    ap.add_argument("--out", default="outputs/calib_check")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root)
    fids = seq.frame_ids
    frames = args.frames or [fids[0], fids[len(fids) // 2], fids[-1]]
    out = Path(args.out) / seq.root.name
    out.mkdir(parents=True, exist_ok=True)

    stats = {}
    for f in frames:
        bodies = seq.bodies(f)
        tiles, names = [], []
        for n in seq.camera_names:
            img = cv2.cvtColor(seq.image(n, f), cv2.COLOR_RGB2BGR)
            stats.setdefault(n, []).append(draw_bodies(img, seq.camera(n), bodies))
            tiles.append(img)
            names.append(n)
        path = out / f"frame_{f:08d}.jpg"
        cv2.imwrite(str(path), contact_sheet(tiles, names), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"frame {f}: {len(bodies)} bodies -> {path}")

    summary = {n: round(float(np.mean(v)), 3) for n, v in stats.items()}
    (out / "stats.json").write_text(json.dumps(summary, indent=1))
    print("fraction of projected joints inside image, per camera:")
    for n, v in summary.items():
        print(f"  {n}: {v:.2f}")


if __name__ == "__main__":
    main()
