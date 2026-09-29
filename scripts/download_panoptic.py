"""Fetch a short clip of a CMU Panoptic sequence and convert it to the fvv layout.

Streams only the requested frame range out of each HD camera video with ffmpeg
(HTTP range seeking), so a 5 s clip from 31 cameras costs ~1-2 GB, not ~50 GB.

    scripts/run.sh python scripts/download_panoptic.py --seq 170221_haggling_b1 --num-frames 150
"""
import argparse
import io
import json
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fvv.data import Camera, save_cameras

BASE = "http://domedb.perception.cs.cmu.edu/webdata/dataset"
HD_FPS = 30000 / 1001


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def load_calibration(seq: str) -> list[Camera]:
    calib = json.loads(fetch(f"{BASE}/{seq}/calibration_{seq}.json"))
    cams = []
    for c in calib["cameras"]:
        if c["type"] != "hd":
            continue
        w, h = c["resolution"]
        cams.append(Camera(c["name"], np.array(c["K"], float), np.array(c["distCoef"], float),
                           np.array(c["R"], float), np.array(c["t"], float).reshape(3), w, h))
    return cams


def load_poses(seq: str) -> dict[int, list[dict]]:
    """All HD-frame 3D skeletons of the sequence, keyed by frame index."""
    data = fetch(f"{BASE}/{seq}/hdPose3d_stage1_coco19.tar")
    poses = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            name = Path(m.name).name
            if not (m.isfile() and name.startswith("body3DScene_")):
                continue
            frame = int(name.removeprefix("body3DScene_").removesuffix(".json"))
            bodies = json.load(tar.extractfile(m))["bodies"]
            poses[frame] = [{"id": b["id"], "joints": np.array(b["joints19"]).reshape(-1, 4).tolist()}
                            for b in bodies]
    return poses


def pick_start(poses: dict[int, list], num_frames: int) -> int:
    """First frame where the next num_frames all contain the max number of people."""
    most = max(len(b) for b in poses.values())
    full = {f for f, b in poses.items() if len(b) == most}
    for f in sorted(full):
        if all(f + i in full for i in range(num_frames)):
            return f
    return min(full)


def extract(seq: str, cam: str, start: int, n: int, out_dir: Path, scale: float) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.jpg"))) >= n:
        return f"{cam}: cached"
    url = f"{BASE}/{seq}/videos/hd_shared_crf20/hd_{cam}.mp4"
    vf = ["-vf", f"scale=iw*{scale}:ih*{scale}:flags=area"] if scale != 1.0 else []
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-ss", f"{start / HD_FPS:.6f}", "-i", url, "-frames:v", str(n), *vf,
           "-q:v", "2", "-start_number", str(start), str(out_dir / "%08d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cam}: {r.stderr.strip()}")
    return f"{cam}: {len(list(out_dir.glob('*.jpg')))} frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="170221_haggling_b1")
    ap.add_argument("--start-frame", type=int, default=None, help="HD frame index; auto if omitted")
    ap.add_argument("--num-frames", type=int, default=150)
    ap.add_argument("--cams", nargs="*", default=None, help="e.g. 00_00 00_05; default all HD cams")
    ap.add_argument("--scale", type=float, default=1.0, help="resize frames (e.g. 0.5 for 960x540)")
    ap.add_argument("--out", default="data/panoptic")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.out) / args.seq
    root.mkdir(parents=True, exist_ok=True)

    cams = load_calibration(args.seq)
    if args.cams:
        cams = [c for c in cams if c.name in args.cams]
    print(f"{len(cams)} HD cameras")

    poses = load_poses(args.seq)
    start = args.start_frame if args.start_frame is not None else pick_start(poses, args.num_frames)
    frames = range(start, start + args.num_frames)
    print(f"frames {frames.start}..{frames.stop - 1} ({args.num_frames / HD_FPS:.1f}s), "
          f"{len(poses.get(start, []))} people at start")

    save_cameras(root / "cameras.json", [c.scaled(args.scale) for c in cams],
                 units="cm", source=f"cmu_panoptic/{args.seq}", fps=HD_FPS)
    (root / "bodies").mkdir(exist_ok=True)
    for f in frames:
        if f in poses:
            (root / "bodies" / f"{f:08d}.json").write_text(json.dumps({"bodies": poses[f]}))

    with ThreadPoolExecutor(args.workers) as ex:
        jobs = [ex.submit(extract, args.seq, c.name, start, args.num_frames,
                          root / "frames" / c.name, args.scale) for c in cams]
        for j in tqdm(jobs, desc="cameras"):
            tqdm.write(j.result())


if __name__ == "__main__":
    main()
