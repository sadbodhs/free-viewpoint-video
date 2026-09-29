"""Video writing and overlays for result videos."""
import subprocess

import cv2
import numpy as np

from fvv.data import Camera


class VideoWriter:
    """H.264 mp4 via ffmpeg (browser-playable). Frames are RGB uint8, even width/height."""

    def __init__(self, path: str, width: int, height: int, fps: float = 30):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.size = (width, height)

    def write(self, rgb: np.ndarray) -> None:
        assert rgb.shape[1::-1] == self.size, (rgb.shape, self.size)
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def label(img: np.ndarray, text: str, pos: str = "top", scale: float = 0.8) -> np.ndarray:
    """Draw text on a dark banner (in place, returns img)."""
    h = int(34 * scale)
    y0 = 0 if pos == "top" else img.shape[0] - h
    img[y0:y0 + h] = (img[y0:y0 + h] * 0.35).astype(np.uint8)
    cv2.putText(img, text, (10, y0 + int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale,
                (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    return img


class MiniMap:
    """Top-down map of the rig with the current virtual camera. Axes: the two directions
    orthogonal to world 'down', so it works for any rig orientation."""

    def __init__(self, cams: list[Camera], down: np.ndarray, size: int = 220):
        self.cams, self.size = cams, size
        a = np.cross(down, [1.0, 0, 0] if abs(down[0]) < 0.9 else [0, 0, 1.0])
        self.axes = np.stack([a / np.linalg.norm(a), np.cross(down, a / np.linalg.norm(a))])
        P = np.stack([c.center for c in cams]) @ self.axes.T
        self.lo, self.span = P.min(0), (P.max(0) - P.min(0)).max() * 1.15
        self.lo -= (self.span - (P.max(0) - P.min(0))) / 2

    def _px(self, X: np.ndarray) -> tuple[int, int]:
        p = (X @ self.axes.T - self.lo) / self.span * (self.size - 1)
        return int(p[0]), int(self.size - 1 - p[1])

    def draw(self, virtual: Camera, highlight: set[str] = frozenset()) -> np.ndarray:
        m = np.full((self.size, self.size, 3), 30, np.uint8)
        for c in self.cams:
            color = (230, 60, 60) if c.name in highlight else (120, 170, 230)
            cv2.circle(m, self._px(c.center), 4, color, -1, cv2.LINE_AA)
        p = self._px(virtual.center)
        q = self._px(virtual.center + virtual.forward * self.span * 0.12)
        cv2.arrowedLine(m, p, q, (255, 220, 60), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(m, p, 6, (255, 220, 60), -1, cv2.LINE_AA)
        return m


def inset(img: np.ndarray, small: np.ndarray, margin: int = 12) -> np.ndarray:
    """Paste `small` into the top-right corner (in place)."""
    h, w = small.shape[:2]
    img[margin:margin + h, -w - margin:-margin] = small
    return img


def even(img: np.ndarray) -> np.ndarray:
    """Crop to even dimensions (required by yuv420p)."""
    return img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]
