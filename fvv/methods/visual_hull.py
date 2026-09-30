"""Real-time Tier A: GPU visual hull people + static background splat (docs/REALTIME.md, R1).

Per frame (no optimization, no dataset annotations):
1. masks: |live frame - static background rendered from that camera| (in that camera's color)
2. carve: coarse voxels (8 cm) then fine voxels (2 cm) inside occupied coarse cells; a voxel is
   occupied if it falls inside the masks of most cameras that see it (a fraction may miss:
   mask holes where a shirt matches the wall) -- a missing camera simply doesn't carve
3. render: voxels become small Gaussians colored per view from the nearest real cameras that
   see them (occlusion via hull depth maps), composited with the background in one rasterization

Offline inputs: calibration, the layered background splat (fvv/methods/layered_splat.py) and a
floor height; everything per-frame runs on the GPU.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

from fvv.data import Camera, MultiViewSequence
from fvv.data.people import estimate_floor
from fvv.eval import StageTimer
from .gaussian_splat import SH_C0, GaussianSplat, apply_affine


class VisualHull:
    name = "hull"
    single_frame = False

    def __init__(self, coarse_cm: float = 8.0, fine_cm: float = 2.0, height_cm: float = 220.0,
                 mask_thresh: float = 0.08, miss_frac: float = 0.1, min_views: int = 3, k_views: int = 3,
                 ckpt_dir: str | Path | None = None, device: str = "cuda", **_):
        self.coarse, self.fine, self.height = coarse_cm, fine_cm, height_cm
        self.mask_thresh, self.miss_frac, self.min_views, self.k = mask_thresh, miss_frac, min_views, k_views
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.device = device
        self.frame_cache: dict[int, dict] = {}

    # ---- offline setup -----------------------------------------------------------------
    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        self.seq, self.train_cams = seq, train_cams
        self.bg = GaussianSplat(device=self.device)
        self.bg.load(self.ckpt_dir / f"layered_bg_{seq.root.name}.pt")
        self.bg.name = "layered_bg"
        for p in self.bg.params.values():
            p.requires_grad_(False)
        self.cam_index = {c: i for i, c in enumerate(self.bg.cam_names)}
        self.cams = {c: seq.camera(c) for c in train_cams}
        self.viewmats = {c: self.bg.viewmat(cam) for c, cam in self.cams.items()}
        self.Ks = {c: torch.tensor(cam.K, dtype=torch.float32, device=self.device) for c, cam in self.cams.items()}

        # background plates in each camera's own color space, and inverse color transforms
        self.plates, self.to_canon = {}, {}
        with torch.no_grad():
            for c, cam in self.cams.items():
                img, _, _ = self.bg._rasterize(self.viewmats[c][None], self.Ks[c][None], cam.width, cam.height)
                A = self.bg.affine[self.cam_index[c]]
                self.plates[c] = apply_affine(img[0], A).clamp(0, 1)
                M = torch.linalg.inv(A[:, :3])
                self.to_canon[c] = (M, A[:, 3])

        # voxel grid: rig footprint x [floor, floor + height], normalized coordinates
        down, floor = estimate_floor(seq)
        C = np.stack([cam.center for cam in seq.cameras.values()])
        h = C @ down
        horiz = C - np.outer(h, down)
        center = horiz.mean(0)
        radius = 0.85 * np.linalg.norm(horiz - center, axis=1).mean()
        e1 = np.cross(down, [1.0, 0, 0] if abs(down[0]) < 0.9 else [0, 0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(down, e1)
        n_xy, n_h = int(2 * radius / self.coarse), int(self.height / self.coarse)
        u = (np.arange(n_xy) + 0.5) * self.coarse - radius
        v = (np.arange(n_h) + 0.5) * self.coarse
        U, V, Hh = np.meshgrid(u, u, v, indexing="ij")
        pts = center + U[..., None] * e1 + V[..., None] * e2 + (floor - Hh)[..., None] * down
        pts = pts.reshape(-1, 3)
        pts = pts[np.linalg.norm((pts - center) - np.outer((pts - center) @ down, down), axis=1) < radius]
        self.axes = torch.tensor(np.stack([e1, e2, down]), dtype=torch.float32, device=self.device)
        self.coarse_pts = torch.tensor((pts - self.bg.center) / self.bg.scale, dtype=torch.float32, device=self.device)
        s = self.coarse / self.fine
        o = (np.stack(np.meshgrid(*[np.arange(int(s))] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / s - 0.5
        self.child_offsets = torch.tensor(o, dtype=torch.float32, device=self.device) @ self.axes * (self.coarse / self.bg.scale)
        self.fine_n = self.fine / self.bg.scale
        print(f"[hull] {len(self.coarse_pts):,} coarse voxels ({self.coarse} cm), fine {self.fine} cm, "
              f"{len(self.cams)} cameras, floor {floor:.1f} cm")

    # ---- per frame ---------------------------------------------------------------------
    def _project(self, c: str, x: torch.Tensor):
        xc = x @ self.viewmats[c][:3, :3].T + self.viewmats[c][:3, 3]
        uv = xc @ self.Ks[c].T
        return uv[:, :2] / uv[:, 2:].clamp(min=1e-6), xc[:, 2]

    def _carve(self, x: torch.Tensor, P: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Vectorized over cameras: P (V,3,4) projections, masks (V,H,W) bool."""
        V, H, W = masks.shape
        uvw = torch.einsum("vij,nj->vni", P[:, :, :3], x) + P[:, None, :, 3]
        z = uvw[..., 2]
        u = uvw[..., 0] / z.clamp(min=1e-6)
        v = uvw[..., 1] / z.clamp(min=1e-6)
        vis = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = v.long().clamp(0, H - 1) * W + u.long().clamp(0, W - 1)
        inside = masks.view(V, -1).gather(1, idx) & vis
        seen, hits = vis.sum(0), inside.sum(0)
        return (seen >= self.min_views) & (hits >= (1 - self.miss_frac) * seen)

    @torch.no_grad()
    def process(self, images: dict, timer: StageTimer) -> dict:
        """images: cam -> RGB uint8 (H, W, 3), numpy or GPU tensor (live ingest decodes to GPU)."""
        cams = [c for c in images if c in self.cams]
        if not cams:                       # every feed lost this slot: background only
            return {"cams": [], "images": None, "voxels": torch.zeros(0, 3, device=self.device)}
        with timer("upload"):
            u8 = torch.stack([torch.as_tensor(images[c]).to(self.device, non_blocking=True) for c in cams])
        with timer("masks"):
            imgs = u8.float() / 255
            plates = torch.stack([self.plates[c] for c in cams])
            d = (imgs - plates).abs().amax(-1)[:, None]
            m = (F.avg_pool2d(d, 5, 1, 2) > self.mask_thresh).float()
            m = -F.max_pool2d(-F.max_pool2d(m, 7, 1, 3), 7, 1, 3)          # closing: fill small holes
            coarse_m = F.max_pool2d(m, 13, 1, 6)[:, 0] > 0.5               # coarse voxels cover more pixels
            fine_m = F.max_pool2d(m, 3, 1, 1)[:, 0] > 0.5
        with timer("carve"):
            P = torch.stack([self.Ks[c] @ self.viewmats[c][:3] for c in cams])
            coarse = self.coarse_pts[self._carve(self.coarse_pts, P, coarse_m)]
            fine = (coarse[:, None] + self.child_offsets[None]).reshape(-1, 3)
            fine = fine[self._carve(fine, P, fine_m)]
        return {"cams": cams, "images": imgs, "voxels": fine}

    def _colorize(self, x, q, s, cams, images, camera: Camera):
        """View-dependent voxel colors from the real cameras nearest the virtual view (occlusion-tested)."""
        n, dev = len(x), self.device
        ctr = torch.tensor(self.bg.center, dtype=torch.float32, device=dev)
        vc = (torch.tensor(camera.center, dtype=torch.float32, device=dev) - ctr) / self.bg.scale
        target = x.mean(0) if n else torch.zeros(3, device=dev)
        vdir = F.normalize(target - vc, dim=0)
        centers = (torch.tensor(np.stack([self.cams[c].center for c in cams]), dtype=torch.float32,
                                device=dev) - ctr) / self.bg.scale
        cos = F.normalize(target - centers, dim=1) @ vdir
        sel = cos.topk(min(self.k, len(cams))).indices.tolist()
        chosen = [cams[i] for i in sel]
        W, H = self.cams[chosen[0]].width, self.cams[chosen[0]].height
        # one multi-camera depth render of the hull for occlusion tests
        depth, _, _ = rasterization(x, q, s, torch.full((n,), 0.99, device=dev), torch.zeros(n, 3, device=dev),
                                    torch.stack([self.viewmats[c] for c in chosen]),
                                    torch.stack([self.Ks[c] for c in chosen]), W, H,
                                    render_mode="ED", packed=False)
        rgb = torch.zeros(n, 3, device=dev)
        wsum = torch.zeros(n, device=dev)
        for j, (i, c) in enumerate(zip(sel, chosen)):
            uv, z = self._project(c, x)
            ui = uv[:, 0].long().clamp(0, W - 1)
            vi = uv[:, 1].long().clamp(0, H - 1)
            visible = (z > 0) & (z <= depth[j, vi, ui, 0] + 2 * self.fine_n)
            M, b = self.to_canon[c]
            col = (images[i][vi, ui] - b) @ M.T
            w = visible.float() * (float(cos[i]) + 1.01) ** 8
            rgb += col * w[:, None]
            wsum += w
        rgb = (rgb / wsum.clamp(min=1e-6)[:, None]).clamp(0, 1)
        opac = torch.where(wsum > 0, 0.99, 0.0)
        return rgb, opac

    @torch.no_grad()
    def render_view(self, state: dict, camera: Camera, timer: StageTimer) -> np.ndarray:
        x, cams, images = state["voxels"], state["cams"], state["images"]
        n = len(x)
        dev = self.device
        s = torch.full((n, 3), self.fine_n * 0.6, device=dev)
        q = torch.zeros(n, 4, device=dev)
        q[:, 0] = 1
        with timer("color"):
            if n and cams:
                rgb, opac = self._colorize(x, q, s, cams, images, camera)
            else:                          # no people / no feeds: background only
                rgb, opac = torch.zeros(0, 3, device=dev), torch.zeros(0, device=dev)
        with timer("rasterize"):
            bg = self.bg.params
            K = torch.tensor(camera.K, dtype=torch.float32, device=dev)[None]
            img, _, _ = rasterization(
                torch.cat([x, bg["means"]]), torch.cat([q, bg["quats"]]), torch.cat([s, torch.exp(bg["scales"])]),
                torch.cat([opac, torch.sigmoid(bg["opacities"])]),
                torch.cat([torch.cat([((rgb - 0.5) / SH_C0)[:, None], torch.zeros(n, 15, 3, device=dev)], 1),
                           torch.cat([bg["sh0"], bg["shN"]], 1)]),
                self.bg.viewmat(camera)[None], K, camera.width, camera.height, sh_degree=3, packed=False)
        with timer("to_cpu"):
            return (img[0].clamp(0, 1) * 255).byte().cpu().numpy()

    # ---- offline-protocol interface (same held-out evaluation as every other method) ----
    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        if frame not in self.frame_cache:
            self.frame_cache = {frame: self.process(
                {c: self.seq.image(c, frame) for c in self.seq.valid_cameras(frame, self.train_cams)}, timer)}
        return self.render_view(self.frame_cache[frame], camera, timer)
