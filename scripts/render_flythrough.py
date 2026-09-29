"""Free-viewpoint flythrough video along the shared orbit path, with a rig mini-map.

Every method renders the same path (fvv.render.orbit_path), so videos are directly comparable.
Static (single-frame) methods give a frozen-time "bullet time" orbit; dynamic ones play time.

    scripts/run.sh python scripts/render_flythrough.py data/panoptic/170221_haggling_b1 --method 3dgs
"""
import argparse
from pathlib import Path

import numpy as np

from fvv.data import MultiViewSequence
from fvv.eval import StageTimer, load_or_create_split
from fvv.methods import METHODS, make_method
from fvv.render import MiniMap, VideoWriter, even, inset, label, orbit_path, scene_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--method", default="3dgs", choices=METHODS)
    ap.add_argument("--frame", type=int, help="frame for single-frame methods (must match a trained checkpoint)")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=240, help="path length in video frames (8 s at 30 fps)")
    ap.add_argument("--out", default="outputs/videos")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    test_cams = load_or_create_split(seq)
    train_cams = [c for c in seq.camera_names if c not in test_cams]
    method = make_method(args.method, ckpt_dir=f"outputs/models/{seq.root.name}")

    if getattr(method, "single_frame", False):
        frame = args.frame if args.frame is not None else next(
            f for f in seq.frame_ids if len(seq.valid_cameras(f)) == len(seq.camera_names))
        times = [frame] * args.n
    else:
        times = [seq.frame_ids[k % len(seq.frame_ids)] for k in range(args.n)]
    method.fit(seq, train_cams, sorted(set(times)))

    template = seq.camera(train_cams[0])
    path = orbit_path(seq, template, n=args.n)
    _, down = scene_frame(seq)
    minimap = MiniMap([seq.camera(c) for c in seq.camera_names], down)

    out = Path(args.out) / seq.root.name
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"flythrough_{args.method}.mp4"
    method.render(times[0], path[0], StageTimer())          # warm-up (CUDA init) outside timing
    timer = StageTimer()
    W, H = even(np.zeros((template.height, template.width))).shape[::-1]
    with VideoWriter(str(video), W, H) as vw:
        for cam, t in zip(path, times):
            with timer("render"):
                img = method.render(t, cam, timer)
            img = even(img.copy())
            inset(img, minimap.draw(cam, highlight=set(test_cams)))
            ms = timer.times["render"][-1]
            label(img, f"{args.method}  |  frame {t}  |  {ms:.0f} ms/frame", pos="bottom")
            vw.write(img)
    lat = timer.summary()["render"]
    print(f"{video}  (render mean {lat['mean_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
