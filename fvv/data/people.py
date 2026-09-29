"""People as 3D capsules around skeleton bones: image masks, point sampling, bone motion.

Skeletons come from the dataset here; for other captures any multi-view 3D pose estimator
(2D keypoints triangulated with the calibrated cameras) provides the same input.
"""
import cv2
import numpy as np

from . import COCO19_EDGES
from .camera import Camera


def bones(bodies: list[dict], edges=COCO19_EDGES) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """{(body_id, edge_idx): (a, b)} for bones whose both joints are observed."""
    out = {}
    for b in bodies:
        J = b["joints"]
        for e, (i, j) in enumerate(edges):
            if J[i, 3] > 0 and J[j, 3] > 0:
                out[(b["id"], e)] = (J[i, :3], J[j, :3])
    return out


def capsule_mask(cam: Camera, segs: list[tuple[np.ndarray, np.ndarray]], radius: float) -> np.ndarray:
    """(H, W) bool mask of capsules (bone segments with `radius`, world units) seen by `cam`."""
    m = np.zeros((cam.height, cam.width), np.uint8)
    for a, b in segs:
        uv, depth = cam.project(np.stack([a, b]), distort=False)
        if (depth <= 0).any():
            continue
        r_px = cam.K[0, 0] * radius / depth.min()
        p, q = (tuple(np.round(x).astype(int)) for x in uv)
        cv2.line(m, p, q, 1, thickness=max(1, int(2 * r_px)), lineType=cv2.LINE_8)
        for pt in (p, q):
            cv2.circle(m, pt, max(1, int(r_px)), 1, -1)
    return m.astype(bool)


def sample_capsules(segs: list[tuple[np.ndarray, np.ndarray]], radius: float, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """n points uniformly inside the union of capsules (approximately; proportional to length)."""
    a = np.stack([s[0] for s in segs])
    b = np.stack([s[1] for s in segs])
    length = np.linalg.norm(b - a, axis=1) + radius
    k = rng.choice(len(segs), n, p=length / length.sum())
    t = rng.uniform(-0.15, 1.15, n)[:, None]
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return a[k] + t * (b[k] - a[k]) + d * radius * rng.uniform(0, 1, (n, 1)) ** (1 / 3)


def bone_motion(prev: dict, cur: dict) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rigid transform per bone between two frames: {key: (R, a_prev, a_cur)}, x' = R (x - a_prev) + a_cur.

    R is the minimal rotation taking the previous bone direction to the current one.
    """
    out = {}
    for k in prev.keys() & cur.keys():
        (a0, b0), (a1, b1) = prev[k], cur[k]
        u = (b0 - a0) / (np.linalg.norm(b0 - a0) + 1e-9)
        v = (b1 - a1) / (np.linalg.norm(b1 - a1) + 1e-9)
        axis = np.cross(u, v)
        s, c = np.linalg.norm(axis), float(u @ v)
        if s < 1e-8:
            R = np.eye(3)
        else:
            axis /= s
            R, _ = cv2.Rodrigues(axis * np.arctan2(s, c))
        out[k] = (R, a0, a1)
    return out
