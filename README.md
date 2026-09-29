# Free-Viewpoint Video

Render a multi-camera recording from any virtual viewpoint, working toward real time.

Roadmap:
0. **Data layer + held-out-camera evaluation** ✓
1. **Offline 3D Gaussian Splatting quality baseline** ✓
2. **Static background / dynamic people (layered, tracked 3D video)** ← current
3. Real-time GPU visual hull + view-dependent texturing
4. Real-time neural rendering (feed-forward / temporal Gaussians)
5. Small fast objects (ball/shuttle): detect, triangulate, render synthetically
6. Streaming server + interactive web viewer
7. Own capture rig (sync, calibration)
8. Generalization: uncalibrated/moving cameras, sparse views

How the offline phases turn into a live system (latency budget, milestones R0–R6):
[docs/REALTIME.md](docs/REALTIME.md).

## Environment

Everything runs in Docker (nothing installed on the host). Needs an NVIDIA GPU + nvidia-container-toolkit.

```bash
scripts/build.sh                 # builds fvv:latest, writes docker/requirements.lock
scripts/run.sh python ...        # runs a command in the container, repo mounted at /workspace
```

## Phase 0 quickstart

```bash
# 5 s clip from all 31 HD cameras of CMU Panoptic (streams only the needed frames)
scripts/run.sh python scripts/download_panoptic.py --seq 170221_haggling_b1 --num-frames 150

# calibration + sync check: projects 3D skeletons into every camera
scripts/run.sh python scripts/check_calibration.py data/panoptic/170221_haggling_b1

# rig layout with held-out cameras
scripts/run.sh python scripts/plot_rig.py data/panoptic/170221_haggling_b1

# held-out-camera evaluation (floor baseline)
scripts/run.sh python scripts/evaluate.py data/panoptic/170221_haggling_b1 --method nearest_view
```

## Phase 1: Gaussian splatting (single frame)

```bash
# sparse points from the training cameras (COLMAP, known poses) -> <seq>/points/<frame>.ply
scripts/run.sh python scripts/triangulate_points.py data/panoptic/170221_haggling_b1

# train 3DGS on one frame + held-out eval; checkpoint and .ply in outputs/models/
scripts/run.sh python scripts/evaluate.py data/panoptic/170221_haggling_b1 --method 3dgs
```

## Result videos

Every method gets the same two videos:

```bash
# flythrough on the shared orbit path, with a rig mini-map (red = held-out cameras)
scripts/run.sh python scripts/render_flythrough.py data/panoptic/170221_haggling_b1 --method 3dgs

# held-out comparison grid: ground truth | method A | method B, per-frame PSNR
scripts/run.sh python scripts/make_comparison_video.py data/panoptic/170221_haggling_b1 \
    --methods nearest_view 3dgs
```

## Layout

```
fvv/data/     Camera, MultiViewSequence (dataset-agnostic on-disk format, see sequence.py)
fvv/eval/     PSNR/SSIM/LPIPS, held-out-camera protocol, per-stage latency timer
fvv/methods/  novel-view methods (fit + render interface in fvv/eval/protocol.py)
fvv/geometry/ COLMAP triangulation with known poses
fvv/render/   shared virtual camera paths, video writer, overlays
scripts/      dataset download, checks, evaluation
docker/       Dockerfile + requirements
```

Data and outputs live in `data/` and `outputs/` (git-ignored).

## Evaluation protocol

Three spread-out cameras (farthest-point sampling on camera centers) are held out. A method
sees only the remaining cameras and renders the held-out viewpoints; the real images are ground
truth. Reported: PSNR / SSIM / LPIPS and per-stage latency (ms, CUDA-synchronized).
