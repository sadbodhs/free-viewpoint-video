"""Pinhole camera with OpenCV-style distortion. World->camera: x_c = R @ X + t."""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Camera:
    name: str
    K: np.ndarray       # (3, 3)
    dist: np.ndarray    # (5,) k1 k2 p1 p2 k3
    R: np.ndarray       # (3, 3) world -> camera
    t: np.ndarray       # (3,)   world -> camera
    width: int
    height: int
    _undistort_maps: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def center(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction (camera +z) in world coordinates."""
        return self.R[2]

    @property
    def world_to_cam(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    def to_cam(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray, distort: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Project world points (N, 3) -> pixels (N, 2) and camera depth (N,)."""
        X = np.ascontiguousarray(X, dtype=np.float64).reshape(-1, 3)
        depth = self.to_cam(X)[:, 2]
        rvec, _ = cv2.Rodrigues(self.R)
        dist = self.dist if distort else np.zeros(5)
        uv, _ = cv2.projectPoints(X, rvec, self.t, self.K, dist)
        return uv.reshape(-1, 2), depth

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Remove lens distortion, keeping K unchanged (use `undistorted()` for the matching camera)."""
        if self._undistort_maps is None:
            self._undistort_maps = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.K, (self.width, self.height), cv2.CV_16SC2)
        return cv2.remap(img, *self._undistort_maps, cv2.INTER_LINEAR)

    def undistorted(self) -> "Camera":
        return Camera(self.name, self.K.copy(), np.zeros(5), self.R, self.t, self.width, self.height)

    def scaled(self, s: float) -> "Camera":
        """Same camera for an image resized by factor s."""
        K = self.K.copy()
        K[:2] *= s
        return Camera(self.name, K, self.dist, self.R, self.t,
                      round(self.width * s), round(self.height * s))
