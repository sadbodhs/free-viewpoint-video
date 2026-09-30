# Optimization plan: less GPU, more accuracy

Study of the current system (RTX 3090, 27 cameras at 960×540, 30 fps) and a prioritized plan.
All "today" numbers are measured; "target" numbers are estimates to be verified by the same
harnesses (`scripts/live_replay.py`, `scripts/evaluate.py`).

## 1. Where the cost is today

**Live mode, per frame (shared by all viewers)** — from `outputs/live/.../hull_clean.json`:

| Stage | ms | Why it costs what it does |
|---|---|---|
| sync + health | **6.4** | Python loop over 27 cameras, each with several GPU→CPU syncs (`.item()`, `torch.equal`) |
| masks | **8.2** | full-resolution float32 ops on 27×960×540: diff, 5×5 blur, 7×7 closing, 13×13 + 3×3 dilations |
| carve | 1.8 | fine already (vectorized) |
| (upload) | 0.2 | frames already on GPU |
| **subtotal** | **~16.5** | **2/3 of it is two "boring" stages** |

**Per viewer:**

| Stage | ms | Why |
|---|---|---|
| color | 1.5 | 3-camera depth render + per-voxel sampling |
| rasterize | 3.5 | **1M static background Gaussians, SH degree 3, re-rendered every frame** |
| JPEG encode | ~3 (CPU) | OpenCV on CPU; ~1.2 CPU cores total; MJPEG is ~10× bigger than H.264 |

**Measured load:** live, 1 viewer ≈ 64% GPU / ~290 W; layered ≈ 40%; idle 0% but **~12 GB held**
(6.3 GB = the 5 s clip preloaded as raw frames). Capacity ≈ live processing + 2–3 viewers.

**Offline (layered 3D video):** ~55 GPU-min per 5 s clip (background ~10 min, people tracking
~16 s/frame × 150), people states 1.05 GB per 5 s (7 MB/frame).

**Accuracy today (held-out cameras, color-corrected PSNR / LPIPS):** live hull 24.5 / 0.27,
offline layered 27.6 / 0.20. Known live failure: missing body parts where gray clothes match the
gray background (mask holes); voxel colors are blocky (one color per 2 cm voxel).

## 2. Efficiency: what to change

| # | Change | Expected effect | Effort |
|---|---|---|---|
| E1 | **Batch health checks**: one stacked thumbnail tensor for all cameras; std / freeze / sharpness computed together, one GPU→CPU sync per frame | 6.4 → **<1 ms** | S |
| E2 | **Masks at lower resolution + fp16/uint8**: coarse carve needs ¼-res masks, fine carve ½-res; separable pooling; fuse with `torch.compile` (or one Triton kernel) | 8.2 → **~1–2 ms** | S |
| E3 | **Compact the static background splat**: prune low-importance Gaussians (1M → ~300k, LightGaussian-style), SH degree 3 → 1 for walls/floor | rasterize 3.5 → **~1.5 ms**/viewer; model 236 MB → ~50 MB | M |
| E4 | **GPU encoding**: nvJPEG (`torchvision.io.encode_jpeg` on CUDA) now; NVENC H.264 via WebRTC later | CPU ~1.2 cores → ~0.2; bandwidth **÷5–10** with H.264 | S / M |
| E5 | **Surface voxels only**: drop interior hull voxels before coloring/rendering | fewer Gaussians (~3–5×): color + raster faster | S |
| E6 | **Multi-viewer batching**: render all viewers in one gsplat call (multi-camera) | lower per-viewer overhead | S |
| E7 | **Don't redo static work**: when a viewer's camera hasn't moved and time is paused, reuse the frame; cache background render when only people change and the camera is still | idle-ish viewers ≈ free | S |
| E8 | **Keep frames compressed in memory** (JPEG/H.264) and decode per slot with nvJPEG/NVDEC — also the realistic path for RTSP | GPU memory 12 GB → **~2 GB**; unlocks long clips | M |
| E9 | **CUDA graphs / fewer kernel launches** for the fixed-shape per-frame path; overlap processing of frame t+1 with rendering of frame t on separate CUDA streams | less launch overhead; lower latency | M |
| E10 | **Offline layered speed**: fused SSIM kernel, fewer tracking steps thanks to skeleton warm start (500 → ~150–200, verify quality curve), lower-res tracking, 250k → ~150k people Gaussians, store per-frame deltas in fp16/quantized | 55 → **~10–15 GPU-min** per 5 s; storage 7 → ~1–2 MB/frame | M |

**Target after E1–E7:** shared processing ~16.5 → **~5 ms**, per viewer ~5 → **~2.5 ms**,
i.e. live with 1 viewer ≈ **20–25% GPU** (from 64%) and **~8–10 viewers per 3090**.

## 3. Accuracy: what to change

| # | Change | Expected effect | Effort |
|---|---|---|---|
| A1 | **Honest metrics first**: add people-only PSNR/LPIPS (inside person masks) and a temporal flicker metric; today's PSNR is dominated by the static background | measures what viewers look at | S |
| A2 | **Better person masks**: fuse background difference with a light person-segmentation network (YOLO-seg / RTMDet-Ins class, TensorRT fp16, batched at 384 px, ~3–5 ms for 27 cams) + shadow suppression (shadows darken but keep chromaticity) | **no missing limbs**; est. +1.5–3 dB on people | M |
| A3 | **Per-pixel (image-based) texturing** instead of one color per voxel: render hull depth from the virtual view, then sample the nearest source cameras per pixel with visibility-aware blending weights | much sharper people (LPIPS ↓ a lot); ~1–2 ms | M |
| A4 | **Photo-consistency carving**: remove hull voxels whose colors disagree across the cameras that see them (hulls are fat in concavities, e.g. between arms and body) | tighter shapes, fewer ghost colors | M |
| A5 | **Temporal stability**: carve with the previous frame's occupancy as a prior; smooth colors over time | less flicker between frames | S |
| A6 | **Background quality**: more key frames and a depth/opacity regularizer for the static splat (removes residual haze near the floor) — helps every mode | + background PSNR | S |
| A7 | **Tier B feed-forward Gaussians (R4)**: network predicts people Gaussians from the 2–4 source views nearest the viewer; distil from the offline layered results; TensorRT fp16 | approach offline quality (~27 dB) live; costs more GPU than the hull (~15–25 ms/viewer) → use when quality matters | L |

Note on the trade-off: A7 improves accuracy but *adds* GPU per viewer; E1–E7 must land first so
there is budget for it. A2 uses the TensorRT/Triton tooling already on the 3090 (`rt_vs_triton`).

## 4. Order of work

| Step | Contents | Why this order |
|---|---|---|
| 0 | **Profiling baseline**: `torch.profiler` / Nsight Systems trace of one live frame; per-stage CUDA-event timings in the replay report | nvidia-smi "utilization" is coarse; optimize against real kernel time |
| 1 | **Quick wins**: E1, E2, E4 (nvJPEG), E5, E7 | ~1 day; biggest GPU drop for least risk |
| 2 | **Metrics**: A1 | so accuracy changes are measured on people, not walls |
| 3 | **Accuracy, live**: A2 masks → A3 per-pixel texturing → A5 temporal | the three visible problems today: holes, blockiness, flicker |
| 4 | **Scale**: E3 compact background, E6 batching, E8 compressed frames | more viewers, long clips, real RTSP input |
| 5 | **Offline speed**: E10 | cheaper to build high-quality clips (longer videos) |
| 6 | **Quality tier**: A7 feed-forward Gaussians, E9 CUDA graphs | once there is GPU budget to spend |

Each step is accepted only with before/after numbers from the same replay (ms per stage, GPU %,
dropped slots, viewers sustained) and evaluation (held-out + people-only metrics).
