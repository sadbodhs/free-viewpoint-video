# Real-time plan

Goal: a viewer moves a virtual camera around a **live** multi-camera capture and sees the
rendered view with broadcast-style latency (≤ 0.5 s glass-to-glass) at 30 fps.

## Where we are (measured on one RTX 3090, 960×540)

| Stage | Today | Real-time budget | Gap |
|---|---|---|---|
| Render (static bg + people, ~1.25M Gaussians) | **3.5–5 ms** | ≤ 8 ms | ✅ already fine |
| Build people for a new frame (layered tracking, 500 steps) | ~10 s | ≤ 20 ms | ❌ ~500× too slow |
| Background model | ~10 min, once | offline | ✅ offline by design |
| Person location (3D skeletons) | from dataset | ≤ 5 ms | needs live pose |

Rendering is solved. The only hard problem is **producing the people's 3D representation for
each new frame in milliseconds** instead of seconds. Per-frame optimization cannot get there;
it has to become a direct computation (visual hull) or a single network pass (feed-forward).

## Architecture: offline once, online per frame

```
OFFLINE (before the event)                 ONLINE (every frame, pipelined)
─────────────────────────                 ─────────────────────────────────────────────────
calibration + sync check  ──┐             cameras ─► decode (NVDEC) ─► person masks ─► 3D people
static background splat  ───┼──► shared ─►                                   │           │
per-camera color affines ───┘    state     3D pose (2D keypoints + triangulation) ─┘           │
                                                                                               ▼
                                     viewer camera pose ─► compose people + static bg ─► rasterize
                                                                                               │
                                                                     encode (NVENC) ─► WebRTC ─► viewer
```

The offline/online split is the same **static background + dynamic people** layering built in
Phase 2 (`fvv/methods/layered_splat.py`): the background never changes during the event, so
only people (and a ball) are computed live.

### Per-frame latency budget (30 fps → 33 ms per stage, stages pipelined)

| # | Stage | Method | Target |
|---|---|---|---|
| 1 | Decode N streams | NVDEC, hardware | 2–5 ms |
| 2 | Person masks | difference vs. rendered static background (+ light segmentation net fallback) | 2–4 ms |
| 3 | 3D pose (optional prior) | batched 2D keypoints (RTMPose-class) + triangulation | 3–5 ms |
| 4 | People geometry | **Tier A:** GPU visual hull · **Tier B:** feed-forward Gaussians | 5–20 ms |
| 5 | Render | gsplat rasterization, people + static background | 4–6 ms |
| 6 | Encode + send | NVENC + WebRTC | 3–5 ms |

Each stage runs on its own CUDA stream / worker, so throughput is 30 fps even though a frame
spends ~40–60 ms in the pipeline; network adds the rest of the latency.

## Two tiers for the people geometry

**Tier A — GPU visual hull (Phase 3, the first live system).**
Carve a voxel grid (128³–256³ over the people's bounding boxes) with the per-camera masks,
extract surface voxels, color them by blending the 2–3 real cameras nearest the virtual camera
(view-dependent texturing). No learning, deterministic, a few ms. Looks "blobby" on fine detail,
but it is the guaranteed real-time fallback and the baseline for Tier B.

**Tier B — feed-forward Gaussians (Phase 4, the quality system).**
A network maps the current frame's images from the source cameras nearest the virtual camera
to people Gaussians in one pass (GPS-Gaussian / MVSplat family: ~20–40 ms on a 3090).
Training data comes from what we already have: held-out-camera supervision on Panoptic, and
the offline layered results as extra targets. Temporal smoothing across frames removes flicker.

## Milestones

| | Milestone | Done when |
|---|---|---|
| R0 | **Live replay harness**: recorded multi-view video fed frame-by-frame at 30 fps with a hard deadline; per-stage ms and dropped-frame counts | we can measure any method "as if live" |
| R1 | **Tier A live**: masks → visual hull → texture → composite with static background splat | ≥ 30 fps on replay with 8–16 cameras, held-out PSNR reported |
| R2 | **Interactive serving**: browser sends camera pose over WebSocket, server renders and streams via WebRTC; camera constrained to well-covered viewpoints | a phone can orbit the replay live |
| R3 | **Live pose + masks** replace dataset skeletons | no dataset-provided annotations used online |
| R4 | **Tier B**: feed-forward people Gaussians | beats Tier A on held-out PSNR/LPIPS at ≥ 30 fps |
| R5 | **Own rig**: 4–8 synced cameras, calibration, live capture | end-to-end live demo |
| R6 | **Small fast objects**: ball/shuttle detected, triangulated, rendered synthetically | ball visible and correctly placed at any viewpoint |

## What carries over from the offline phases

- Calibration, sync checks, dropped-frame handling (`fvv/data`).
- Static background splat + per-camera color affines (`layered_bg` checkpoint).
- Held-out-camera evaluation with latency timing (`fvv/eval`), shared flythrough path and
  comparison videos (`fvv/render`) — every real-time method is scored the same way.
- Offline layered results become the quality reference Tier A/B are compared against, and
  training targets for Tier B.

## Robustness with live (RTSP) feeds

**Principle: never wait for a late camera.** Frames are assembled into fixed 33 ms slots by
*capture* timestamp; whatever is missing or late for a slot is invalid for that slot only
(the live version of `invalid_frames` / `valid_cameras`). Every method must work with a
different camera subset each frame.

| Issue | Effect | Handling |
|---|---|---|
| Camera down / stream stalls | fewer views | health monitor (arrival gaps, decode errors, frozen/blank frames); visual hull treats a missing camera as "don't carve", feed-forward picks nearest *available* views; allowed viewer region shrinks around the gap; reconnect with backoff, re-verify calibration before reuse |
| RTP jitter | out-of-sync views | per-stream jitter buffer (50–150 ms), match by RTP→NTP capture time (RTCP SR), cameras on PTP/NTP |
| Resolution / fps change mid-stream | decoder reinit, intrinsics mismatch | reinit decoder, rescale K (`Camera.scaled`); aspect/crop change → recalibrate |
| Packet loss | blocky/smeared frames until next I-frame | mark invalid until IDR; request GOP ≤ 1 s, no B-frames (latency) |
| Auto exposure / WB / focus | color & sharpness drift | lock on camera; online per-camera color affine vs. background render |
| Motion blur / defocus | soft sources | sharpness score (Laplacian variance) → lower weight in texturing/selection |
| Rolling shutter | fast objects skewed | global-shutter cameras for the rig; else per-row time correction |
| Camera bumped / drifts | misaligned view | live frame vs. background rendered from that camera; error jump → re-estimate pose (PnP against background) |
| Lighting change, glare, dirt, rain, occluding spectator | one camera silently bad | per-camera error vs. background render → demote; online background color update; shutter set against 50/60 Hz flicker |
| Transport | loss vs. delay | RTSP-over-TCP or SRT; multicast for multiple consumers; isolated camera network, credentials in a secret store |

**Adding cameras dynamically.** Intrinsics once per camera+lens (avoid zoom/PTZ unless it
reports zoom). Pose without a board: match the new view against renders of the static
background splat → PnP + RANSAC → refine (court lines where available). Time offset via
PTP/NTP or motion/audio cross-correlation; color affine fit in seconds. A new camera is on
probation until reprojection and cross-camera checks pass.

**Zooming past the source resolution.** Cap zoom adaptively from the best source camera's
pixel density at the look-at point (~1.5–2×), anti-aliased (Mip-Splatting-style) rasterization,
blend toward the real camera image when the view ray aligns with one, render the zoomed crop
at higher internal resolution; generative super-resolution only as an optional, labelled mode.

**Capacity (estimates, RTX 3090; R0 measures the real numbers).** NVDEC ~15–25 streams of
1080p30; masks ~0.3–1 ms/camera; visual hull scales with voxels × cameras (16 cams × 256³ ≈ a
few ms); feed-forward people cost is per *viewer* (uses 2–4 nearest cameras), not per camera;
render ~5 ms/viewer; GeForce NVENC limits concurrent encode sessions (data-center GPUs don't).
One 3090 ≈ 8–16 cameras + a few viewers; beyond that split ingest nodes (~16 cams/GPU) from
render nodes (~5 viewers/GPU) or render client-side.

**R0 fault injection.** The replay harness can drop cameras, add jitter, change resolution,
blur/freeze streams and corrupt frames, so every method is scored under failures too.

## Scaling to many viewers

- **Server-side rendering first**: ~5 ms per view → ~5 concurrent viewers per 3090 at 30 fps,
  more with lower resolution or batching views.
- **Client-side later**: stream the static background once (compressed splat), then only the
  people layer per frame (quantized, ~0.3–1 MB/frame); the browser renders any view itself.
  This scales to unlimited viewers and enables AR placement (WebXR).
