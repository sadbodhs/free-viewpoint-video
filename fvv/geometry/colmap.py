"""Sparse point cloud from calibrated cameras via COLMAP (known poses: triangulate only, no SfM).

Follows COLMAP's "reconstruct sparse model from known camera poses" recipe:
extract + match SIFT, overwrite database intrinsics with ours, write the known poses as a
text model, then run point_triangulator.
"""
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fvv.data import MultiViewSequence


def _colmap(*args: str) -> None:
    r = subprocess.run(["colmap", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"colmap {args[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def triangulate(seq: MultiViewSequence, frame: int, cams: list[str],
                workdir: str | Path | None = None, peak_threshold: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Return (points (N, 3) in world units, colors (N, 3) uint8) for one frame.

    `seq` should be undistorted (pinhole) so COLMAP's PINHOLE model matches the images.
    peak_threshold below COLMAP's default (0.0067) finds more features in low-texture scenes.
    """
    tmp = tempfile.TemporaryDirectory() if workdir is None else None
    work = Path(workdir or tmp.name)
    img_dir, db = work / "images", work / "database.db"
    known, tri = work / "known", work / "triangulated"
    for d in (img_dir, known, tri):
        d.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)

    for c in cams:
        cv2.imwrite(str(img_dir / f"{c}.jpg"), cv2.cvtColor(seq.image(c, frame), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    _colmap("feature_extractor", "--database_path", str(db), "--image_path", str(img_dir),
            "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0",
            "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_num_features", "16384",
            "--SiftExtraction.peak_threshold", str(peak_threshold))
    _colmap("exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", "0")

    # Overwrite the database intrinsics with ours and write the known poses as a text model.
    con = sqlite3.connect(db)
    rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    cam_lines, img_lines = [], []
    for image_id, name, camera_id in rows:
        cam = seq.camera(Path(name).stem)
        fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
        params = np.array([fx, fy, cx, cy], np.float64)
        con.execute("UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 "
                    "WHERE camera_id=?", (cam.width, cam.height, params.tobytes(), camera_id))
        cam_lines.append(f"{camera_id} PINHOLE {cam.width} {cam.height} {fx} {fy} {cx} {cy}")
        qx, qy, qz, qw = Rotation.from_matrix(cam.R).as_quat()
        tx, ty, tz = cam.t
        img_lines.append(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n")
    con.commit()
    con.close()
    (known / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (known / "images.txt").write_text("\n".join(img_lines) + "\n")
    (known / "points3D.txt").write_text("")

    _colmap("point_triangulator", "--database_path", str(db), "--image_path", str(img_dir),
            "--input_path", str(known), "--output_path", str(tri),
            "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0")
    _colmap("model_converter", "--input_path", str(tri), "--output_path", str(tri), "--output_type", "TXT")

    pts, cols = [], []
    for line in (tri / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        err, track_len = float(v[7]), (len(v) - 8) // 2
        if err < 2.0 and track_len >= 3:            # reprojection error (px) and views per point
            pts.append([float(x) for x in v[1:4]])
            cols.append([int(x) for x in v[4:7]])
    if tmp:
        tmp.cleanup()
    return np.array(pts, np.float64).reshape(-1, 3), np.array(cols, np.uint8).reshape(-1, 3)


def save_points_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, a in enumerate("xyz"):
        data[a] = pts[:, i]
    for i, a in enumerate(("red", "green", "blue")):
        data[a] = cols[:, i]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def load_points_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    body = raw[raw.index(b"end_header\n") + len(b"end_header\n"):]
    data = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    return (np.stack([data[a] for a in "xyz"], 1).astype(np.float64),
            np.stack([data[a] for a in ("red", "green", "blue")], 1))
