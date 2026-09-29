"""Dataset-agnostic multi-view sequence.

On-disk layout (every dataset is converted to this):
    <root>/cameras.json               {"cameras": [{name, K, dist, R, t, width, height}], "units": ...}
    <root>/frames/<cam>/<frame:08d>.jpg
    <root>/bodies/<frame:08d>.json    optional 3D skeletons: {"bodies": [{"id", "joints": [[x,y,z,conf],...]}]}
    <root>/excluded_cameras.json      optional {"cam": "reason"}; skipped unless cams are given explicitly
    <root>/invalid_frames.json        optional {"cam": [frame, ...]}: dropped/blank frames to skip
"""
import json
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera


def load_cameras(path: Path) -> dict[str, Camera]:
    meta = json.loads(Path(path).read_text())
    cams = {}
    for c in meta["cameras"]:
        cams[c["name"]] = Camera(
            name=c["name"], K=np.array(c["K"], float), dist=np.array(c["dist"], float),
            R=np.array(c["R"], float), t=np.array(c["t"], float).reshape(3),
            width=int(c["width"]), height=int(c["height"]))
    return cams


def save_cameras(path: Path, cams: list[Camera], **extra) -> None:
    out = {"cameras": [dict(name=c.name, K=c.K.tolist(), dist=c.dist.tolist(), R=c.R.tolist(),
                            t=c.t.tolist(), width=c.width, height=c.height) for c in cams], **extra}
    Path(path).write_text(json.dumps(out, indent=1))


class MultiViewSequence:
    def __init__(self, root: str | Path, cams: list[str] | None = None,
                 undistort: bool = False, scale: float = 1.0):
        self.root = Path(root)
        all_cams = load_cameras(self.root / "cameras.json")
        excluded_path = self.root / "excluded_cameras.json"
        self.excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else {}
        invalid_path = self.root / "invalid_frames.json"
        self.invalid_frames = ({k: set(v) for k, v in json.loads(invalid_path.read_text()).items()}
                               if invalid_path.exists() else {})
        names = cams or sorted(n for n in all_cams
                               if (self.root / "frames" / n).is_dir() and n not in self.excluded)
        self.cameras = {n: all_cams[n] for n in names}
        self.undistort = undistort
        self.scale = scale

    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras)

    @cached_property
    def frame_ids(self) -> list[int]:
        """Frames present for every selected camera."""
        sets = [{int(p.stem) for p in (self.root / "frames" / n).glob("*.jpg")} for n in self.cameras]
        return sorted(set.intersection(*sets)) if sets else []

    def camera(self, name: str) -> Camera:
        """Camera matching the images returned by `image()` (after undistort/scale)."""
        cam = self.cameras[name]
        if self.undistort:
            cam = cam.undistorted()
        return cam.scaled(self.scale) if self.scale != 1.0 else cam

    def image(self, cam: str, frame: int) -> np.ndarray:
        """RGB uint8 (H, W, 3)."""
        path = self.root / "frames" / cam / f"{frame:08d}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if self.undistort:
            img = self.cameras[cam].undistort(img)
        if self.scale != 1.0:
            c = self.camera(cam)
            img = cv2.resize(img, (c.width, c.height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def frame(self, frame: int, cams: list[str] | None = None) -> dict[str, np.ndarray]:
        return {n: self.image(n, frame) for n in (cams or self.camera_names)}

    def is_valid(self, cam: str, frame: int) -> bool:
        return frame not in self.invalid_frames.get(cam, ())

    def valid_cameras(self, frame: int, cams: list[str] | None = None) -> list[str]:
        return [c for c in (cams or self.camera_names) if self.is_valid(c, frame)]

    def scan_blank_frames(self, min_std: float = 8.0) -> dict[str, list[int]]:
        """Frames that are (near) constant, e.g. dropped frames filled with a solid color."""
        blank = {}
        for n in self.camera_names:
            for f in self.frame_ids:
                img = cv2.imread(str(self.root / "frames" / n / f"{f:08d}.jpg"), cv2.IMREAD_REDUCED_COLOR_8)
                if img.reshape(-1, 3).std(0).max() < min_std:
                    blank.setdefault(n, []).append(f)
        return blank

    def bodies(self, frame: int) -> list[dict]:
        """3D skeletons for a frame as [{"id", "joints": (J, 4) array}], empty if unavailable."""
        path = self.root / "bodies" / f"{frame:08d}.json"
        if not path.exists():
            return []
        return [{"id": b["id"], "joints": np.array(b["joints"], float)}
                for b in json.loads(path.read_text())["bodies"]]
