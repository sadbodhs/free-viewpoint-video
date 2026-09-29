"""Held-out-camera evaluation protocol shared by every method.

A method sees only the training cameras and must render the held-out cameras'
viewpoints; the real held-out images are the ground truth.
"""
import json
import shutil
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from fvv.data import Camera, MultiViewSequence
from .metrics import all_metrics
from .timing import StageTimer


class NovelViewMethod(Protocol):
    name: str

    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        """One-off setup (e.g. background reconstruction). Not counted as per-frame latency."""

    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        """Render RGB uint8 (H, W, 3) for `camera` at `frame`, timing stages with `timer`."""


def pick_test_cameras(seq: MultiViewSequence, n: int = 3) -> list[str]:
    """Farthest-point sampling on camera centers: spread-out, deterministic held-out views."""
    names = seq.camera_names
    C = np.stack([seq.cameras[k].center for k in names])
    chosen = [int(np.argmin(np.linalg.norm(C - C.mean(0), axis=1)))]
    while len(chosen) < n:
        d = np.min(np.linalg.norm(C[:, None] - C[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return sorted(names[i] for i in chosen)


def load_or_create_split(seq: MultiViewSequence, n: int = 3) -> list[str]:
    """Held-out cameras are fixed per sequence (saved in split.json) so results stay comparable."""
    path = seq.root / "split.json"
    if path.exists():
        return json.loads(path.read_text())["test_cams"]
    test = pick_test_cameras(seq, n)
    path.write_text(json.dumps({"test_cams": test}))
    return test


def evaluate(method: NovelViewMethod, seq: MultiViewSequence, test_cams: list[str],
             frames: list[int], out_dir: str | Path, save_images: int = 3) -> dict:
    out = Path(out_dir) / method.name
    for d in ("images", "pred"):
        shutil.rmtree(out / d, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    train_cams = [c for c in seq.camera_names if c not in test_cams]
    method.fit(seq, train_cams, frames)

    timer = StageTimer()
    rows = []
    for i, f in enumerate(frames):
        for c in seq.valid_cameras(f, test_cams):
            cam = seq.camera(c)
            with timer("total"):
                pred = method.render(f, cam, timer)
            gt = seq.image(c, f)
            rows.append({"frame": f, "cam": c, **all_metrics(pred, gt)})
            # every prediction is kept for the comparison video (scripts/make_comparison_video.py)
            (out / "pred" / c).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "pred" / c / f"{f:08d}.jpg"), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if i < save_images:
                side = np.concatenate([gt, pred], axis=1)
                cv2.imwrite(str(out / "images" / f"{c}_{f:08d}.jpg"),
                            cv2.cvtColor(side, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

    keys = ("psnr", "psnr_cc", "ssim", "lpips")
    result = {
        "method": method.name,
        "train_cams": train_cams,
        "test_cams": test_cams,
        "frames": [frames[0], frames[-1], len(frames)],
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "per_camera": {c: {k: float(np.mean([r[k] for r in rows if r["cam"] == c])) for k in keys}
                       for c in test_cams},
        "latency": timer.summary(),
        "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(result, indent=1))
    return result
