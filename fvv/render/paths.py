"""Virtual camera paths. Every method renders the SAME path so result videos are comparable."""
import numpy as np

from fvv.data import Camera, MultiViewSequence


def look_at(eye: np.ndarray, target: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV-convention (x right, y down, z forward) world->camera rotation and translation."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def scene_frame(seq: MultiViewSequence) -> tuple[np.ndarray, np.ndarray]:
    """(target point, world 'down' direction): people centroid if skeletons exist, else rig center.

    'down' is estimated as the average camera image-down axis, so it works for any rig orientation.
    """
    down = np.mean([c.R[1] for c in seq.cameras.values()], axis=0)
    down /= np.linalg.norm(down)
    joints = [b["joints"][b["joints"][:, 3] > 0, :3] for b in seq.bodies(seq.frame_ids[0])]
    joints = [j for j in joints if len(j)]
    if joints:
        target = np.concatenate(joints).mean(0)
    else:
        target = np.mean([c.center for c in seq.cameras.values()], axis=0)
    return target, down


def orbit_path(seq: MultiViewSequence, template: Camera, n: int = 240, boundary_frac: float = 0.75,
               min_dist: float = 0.35, height_swing: float = 0.35) -> list[Camera]:
    """Full circle around the scene, always looking at it and always staying inside the rig.

    The rig footprint is approximated by a circle fit to the camera centers. At each angle the
    camera sits boundary_frac of the way from the target to that circle (but at least min_dist of
    the rig radius from the target), so off-center subjects never push the path outside the
    captured volume. Height oscillates across the camera rings (height_swing of their span).
    """
    target, down = scene_frame(seq)
    C = np.stack([c.center for c in seq.cameras.values()])
    rel = C - target
    h = rel @ down                                     # signed height below target (negative = above)
    horiz = rel - np.outer(h, down)
    e1 = horiz[0] / np.linalg.norm(horiz[0])
    e2 = np.cross(down, e1)
    # rig footprint circle in the (e1, e2) plane, relative to the target
    P = np.stack([horiz @ e1, horiz @ e2], 1)
    center = P.mean(0)
    rig_r = np.linalg.norm(P - center, axis=1).mean()
    h_mid, h_amp = np.median(h), height_swing * (h.max() - h.min()) / 2

    cams = []
    for k in range(n):
        a = 2 * np.pi * k / n
        d = np.array([np.cos(a), np.sin(a)])
        # distance from target along d to the footprint circle: |s d - center| = R
        b = d @ center
        s = b + np.sqrt(max(b * b - center @ center + rig_r * rig_r, 0.0))
        s = max(boundary_frac * s, min_dist * rig_r)
        eye = target + s * (d[0] * e1 + d[1] * e2) + (h_mid + h_amp * np.sin(2 * a)) * down
        R, t = look_at(eye, target, down)
        cams.append(Camera(f"orbit_{k:04d}", template.K.copy(), np.zeros(5), R, t,
                           template.width, template.height))
    return cams
