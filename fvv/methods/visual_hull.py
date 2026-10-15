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
from gsplat import rasterization, spherical_harmonics

from fvv.data import Camera, MultiViewSequence
from fvv.data.people import estimate_floor
from fvv.eval import StageTimer
from .gaussian_splat import SH_C0, GaussianSplat, apply_affine, compact, prune_near_cameras


class VisualHull:
    name = "hull"
    single_frame = False

    def __init__(self, coarse_cm: float = 8.0, fine_cm: float = 2.0, height_cm: float = 220.0,
                 mask_thresh: float = 0.08, miss_frac: float = 0.1, min_views: int = 3, k_views: int = 3,
                 seg: str | None = None, seg_res: int = 256, seg_hi: float = 0.5, seg_lo: float = 0.2,
                 seg_gate: bool = True, texture: str = "pixel", grid_frac: float = 0.97,
                 bg_name: str = "layered_bg", prune_near_cm: float = 60.0, bg_keep: float = 0.5,
                 bg_sh_degree: int = 3,
                 ckpt_dir: str | Path | None = None, device: str = "cuda", **_):
        self.coarse, self.fine, self.height = coarse_cm, fine_cm, height_cm
        self.mask_thresh, self.miss_frac, self.min_views, self.k = mask_thresh, miss_frac, min_views, k_views
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.device = device
        self.frame_cache: dict[int, dict] = {}
        # optional person segmentation fused with background difference (fills gray-on-gray holes)
        self.seg, self.seg_res, self.seg_hi, self.seg_lo, self.seg_gate = seg, seg_res, seg_hi, seg_lo, seg_gate
        self.texture, self.grid_frac, self.bg_name = texture, grid_frac, bg_name
        self.prune_near_cm, self.bg_keep, self.bg_sh_degree = prune_near_cm, bg_keep, bg_sh_degree
        self.seg_net = self._load_seg(seg) if seg else None

    # ---- offline setup -----------------------------------------------------------------
    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        self.seq, self.train_cams = seq, train_cams
        self.bg = GaussianSplat(device=self.device)
        self.bg.load(self.ckpt_dir / f"{self.bg_name}_{seq.root.name}.pt")
        self.bg.name = "layered_bg"
        for p in self.bg.params.values():
            p.requires_grad_(False)
        down, floor = estimate_floor(seq)
        if self.prune_near_cm:
            n = prune_near_cameras(self.bg, list(seq.cameras.values()), down, floor, self.prune_near_cm)
            print(f"[hull] pruned {n:,} background floaters within {self.prune_near_cm:.0f} cm of cameras")
        if self.bg_keep < 1.0 or self.bg_sh_degree < 3:
            n = compact(self.bg, [seq.camera(c) for c in train_cams], self.bg_keep, self.bg_sh_degree)
            print(f"[hull] compacted background: -{n:,} Gaussians, SH degree {self.bg.sh_degree}")
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
                plate = apply_affine(img[0], A).clamp(0, 1).permute(2, 0, 1)[None]
                self.plates[c] = F.avg_pool2d(plate, 2)[0].half()          # masks run at half resolution
                M = torch.linalg.inv(A[:, :3])
                self.to_canon[c] = (M, A[:, 3])

        # Fixed-shape per-frame tensors over ALL training cameras: missing/unhealthy cameras are masked
        # out instead of dropped, so shapes never change (no allocation spikes; CUDA-graph friendly).
        self.order = list(train_cams)
        self.slot = {c: i for i, c in enumerate(self.order)}
        cam0 = self.cams[self.order[0]]
        self.frame_buf = torch.zeros(len(self.order), cam0.height, cam0.width, 3, dtype=torch.uint8,
                                     device=self.device)
        self.plate_stack = torch.stack([self.plates[c] for c in self.order])
        half = torch.diag(torch.tensor([0.5, 0.5, 1.0], device=self.device))
        self.P_half = torch.stack([half @ self.Ks[c] @ self.viewmats[c][:3] for c in self.order])

        # voxel grid: rig footprint x [floor, floor + height], normalized coordinates
        C = np.stack([cam.center for cam in seq.cameras.values()])
        h = C @ down
        horiz = C - np.outer(h, down)
        center = horiz.mean(0)
        radius = self.grid_frac * np.linalg.norm(horiz - center, axis=1).mean()   # people can stand near the wall
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

    def _load_seg(self, name: str):
        from torchvision.models import segmentation as tvseg
        ctor, weights = {
            "lraspp": (tvseg.lraspp_mobilenet_v3_large, tvseg.LRASPP_MobileNet_V3_Large_Weights),
            "deeplab": (tvseg.deeplabv3_mobilenet_v3_large, tvseg.DeepLabV3_MobileNet_V3_Large_Weights),
        }[name]
        net = ctor(weights=weights.COCO_WITH_VOC_LABELS_V1).to(self.device).eval().half()
        self._seg_norm = (torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1).half(),
                          torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1).half())
        return net

    def _person_prob(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Person probability (N, 1, h, w) at `size` from images x (N, 3, H, W) in [0, 1], fp16."""
        h = self.seg_res
        w = int(h * x.shape[3] / x.shape[2]) // 8 * 8
        mean, std = self._seg_norm
        inp = (F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) - mean) / std
        out = self.seg_net(inp)["out"].float().softmax(1)[:, 15:16]          # VOC class 15 = person
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False).half()

    # ---- per frame ---------------------------------------------------------------------
    def _project(self, c: str, x: torch.Tensor):
        xc = x @ self.viewmats[c][:3, :3].T + self.viewmats[c][:3, 3]
        uv = xc @ self.Ks[c].T
        return uv[:, :2] / uv[:, 2:].clamp(min=1e-6), xc[:, 2]

    def _carve(self, x: torch.Tensor, P: torch.Tensor, masks: torch.Tensor,
               valid: torch.Tensor | None = None) -> torch.Tensor:
        """Vectorized over cameras: P (V,3,4) projections, masks (V,H,W) bool, valid (V,) bool —
        an invalid camera neither sees nor carves (a lost feed must not delete people)."""
        V, H, W = masks.shape
        uvw = torch.einsum("vij,nj->vni", P[:, :, :3], x) + P[:, None, :, 3]
        z = uvw[..., 2]
        u = uvw[..., 0] / z.clamp(min=1e-6)
        v = uvw[..., 1] / z.clamp(min=1e-6)
        vis = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if valid is not None:
            vis &= valid[:, None]
        idx = v.long().clamp(0, H - 1) * W + u.long().clamp(0, W - 1)
        inside = masks.view(V, -1).gather(1, idx) & vis
        seen, hits = vis.sum(0), inside.sum(0)
        return (seen >= self.min_views) & (hits >= (1 - self.miss_frac) * seen)

    def _surface(self, x: torch.Tensor) -> torch.Tensor:
        """Keep voxels with at least one empty neighbour (interior voxels are never seen):
        scatter into a dense grid over their bounding box, erode with a 3x3x3 min-pool."""
        if len(x) == 0:
            return x
        ijk = torch.round((x - x[0]) @ self.axes.T / self.fine_n).long()
        ijk -= ijk.min(0).values - 1                                  # 1-voxel empty border
        D, H, W = (ijk.max(0).values + 2).tolist()
        grid = torch.zeros(1, 1, D, H, W, device=x.device, dtype=torch.half)
        grid[0, 0, ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
        interior = -F.max_pool3d(-grid, 3, 1, 1)[0, 0]
        return x[interior[ijk[:, 0], ijk[:, 1], ijk[:, 2]] < 0.5]

    @torch.no_grad()
    def process(self, images: dict, timer: StageTimer) -> dict:
        """images: cam -> RGB uint8 (H, W, 3), numpy or GPU tensor (live ingest decodes to GPU)."""
        cams = [c for c in images if c in self.cams]
        if not cams:                       # every feed lost this slot: background only
            return {"cams": [], "images": None, "voxels": torch.zeros(0, 3, device=self.device)}
        with timer("upload"):
            for c in cams:
                self.frame_buf[self.slot[c]].copy_(torch.as_tensor(images[c]), non_blocking=True)
            present = set(cams)
            valid = torch.tensor([c in present for c in self.order], device=self.device)
            u8 = self.frame_buf
        with timer("masks"):
            # half resolution, fp16: voxels project to several pixels, so full-res masks are wasted work
            x = F.avg_pool2d(u8.permute(0, 3, 1, 2).half() / 255, 2)
            d = (x - self.plate_stack).abs().amax(1, keepdim=True)
            m = (F.avg_pool2d(d, 3, 1, 1) > self.mask_thresh).half()
            if self.seg_net is not None:
                with timer("segment"):
                    p = self._person_prob(u8.permute(0, 3, 1, 2).half() / 255, m.shape[2:])
                if self.seg_gate:      # background changes far from any person (shadows, noise) are dropped
                    m = m * F.max_pool2d((p > self.seg_lo).half(), 9, 1, 4)
                m = torch.maximum(m, (p > self.seg_hi).half())                # fill holes where the net is sure
            m = -F.max_pool2d(-F.max_pool2d(m, 5, 1, 2), 5, 1, 2)          # closing: fill small holes
            coarse_m = F.max_pool2d(m, 7, 1, 3)[:, 0] > 0.5                # coarse voxels cover more pixels
            fine_m = m[:, 0] > 0.5
        with timer("carve"):
            coarse = self.coarse_pts[self._carve(self.coarse_pts, self.P_half, coarse_m, valid)]
            fine = (coarse[:, None] + self.child_offsets[None]).reshape(-1, 3)
            fine = self._surface(fine[self._carve(fine, self.P_half, fine_m, valid)])
        # images[i] pairs with cams[i] (views into the fixed buffer, no copies)
        return {"cams": cams, "images": [u8[self.slot[c]] for c in cams], "voxels": fine}

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
            col = (images[i][vi, ui].float() / 255 - b) @ M.T
            w = visible.float() * (float(cos[i]) + 1.01) ** 8
            rgb += col * w[:, None]
            wsum += w
        rgb = (rgb / wsum.clamp(min=1e-6)[:, None]).clamp(0, 1)
        opac = torch.where(wsum > 0, 0.99, 0.0)
        return rgb, opac

    def _sources(self, cams, target: torch.Tensor, vc: torch.Tensor, k: int):
        """Indices of the k real cameras whose viewing direction best matches the virtual one."""
        ctr = torch.tensor(self.bg.center, dtype=torch.float32, device=self.device)
        centers = (torch.tensor(np.stack([self.cams[c].center for c in cams]), dtype=torch.float32,
                                device=self.device) - ctr) / self.bg.scale
        cos = F.normalize(target - centers, dim=1) @ F.normalize(target - vc, dim=0)
        return cos.topk(min(k, len(cams))).indices.tolist(), centers

    def _render_pixel(self, state: dict, camera: Camera, timer: StageTimer) -> torch.Tensor:
        """Image-based texturing (unstructured-lumigraph style): render the hull's depth from the
        virtual camera, lift each people pixel to 3D, and blend the real cameras' pixels there with
        visibility tests and angle weights. Sharper than one color per voxel."""
        dev = self.device
        x, cams, images = state["voxels"], state["cams"], state["images"]
        n, W, H = len(x), camera.width, camera.height
        bg = self.bg.params
        vm = self.bg.viewmat(camera)
        K = torch.tensor(camera.K, dtype=torch.float32, device=dev)
        s = torch.full((n, 3), self.fine_n * 0.6, device=dev)
        q = torch.zeros(n, 4, device=dev)
        q[:, 0] = 1
        with timer("rasterize"):
            # One pass over background + hull: RGB carries the background's (already occluded)
            # contribution, a 4th channel the fraction of each pixel the people cover. Haze in front of
            # the people then just dims them, as real compositing would.
            campos = -vm[:3, :3].T @ vm[:3, 3]
            bg_rgb = torch.clamp_min(spherical_harmonics(self.bg.sh_degree, bg["means"] - campos,
                                                         torch.cat([bg["sh0"], bg["shN"]], 1)) + 0.5, 0)
            feats = torch.cat([torch.cat([bg_rgb, torch.zeros(len(bg_rgb), 1, device=dev)], 1),
                               torch.cat([torch.zeros(n, 3, device=dev), torch.ones(n, 1, device=dev)], 1)])
            out, _, _ = rasterization(torch.cat([bg["means"], x]), torch.cat([bg["quats"], q]),
                                      torch.cat([torch.exp(bg["scales"]), s]),
                                      torch.cat([torch.sigmoid(bg["opacities"]), torch.full((n,), 0.99, device=dev)]),
                                      feats, vm[None], K[None], W, H, packed=False)
            img, wp = out[0, ..., :3], out[0, ..., 3]
        if n == 0 or not cams:
            return img.clamp(0, 1)
        with timer("color"):
            hd, _, _ = rasterization(x, q, s, torch.full((n,), 0.99, device=dev), torch.zeros(n, 3, device=dev),
                                     vm[None], K[None], W, H, render_mode="ED", packed=False)
            hd = hd[0, ..., 0]
            ys, xs = torch.nonzero(wp > 0.02, as_tuple=True)
            if len(ys) == 0:
                return img.clamp(0, 1)
            z = hd[ys, xs]
            pix = torch.stack([xs.float() + 0.5, ys.float() + 0.5, torch.ones_like(z)], 1)
            Xw = ((pix @ torch.linalg.inv(K).T) * z[:, None] - vm[:3, 3]) @ vm[:3, :3]   # camera -> world
            vc = campos
            sel, centers = self._sources(cams, x.mean(0), vc, self.k)
            chosen = [cams[i] for i in sel]
            Ws, Hs = self.cams[chosen[0]].width, self.cams[chosen[0]].height
            sd, _, _ = rasterization(x, q, s, torch.full((n,), 0.99, device=dev), torch.zeros(n, 3, device=dev),
                                     torch.stack([self.viewmats[c] for c in chosen]),
                                     torch.stack([self.Ks[c] for c in chosen]), Ws, Hs,
                                     render_mode="ED", packed=False)
            dv = F.normalize(vc - Xw, dim=1)
            rgb = torch.zeros(len(z), 3, device=dev)
            wsum = torch.zeros(len(z), device=dev)
            fallback = torch.zeros(len(z), 3, device=dev)
            tol = 3.0 / self.bg.scale                                          # 3 cm depth tolerance
            for j, (i, c) in enumerate(zip(sel, chosen)):
                Xs = Xw @ self.viewmats[c][:3, :3].T + self.viewmats[c][:3, 3]
                uv = Xs @ self.Ks[c].T
                zc = uv[:, 2]
                uv = uv[:, :2] / zc.clamp(min=1e-6)[:, None]
                ui = uv[:, 0].long().clamp(0, Ws - 1)
                vi = uv[:, 1].long().clamp(0, Hs - 1)
                inb = (uv[:, 0] >= 0) & (uv[:, 0] < Ws) & (uv[:, 1] >= 0) & (uv[:, 1] < Hs) & (zc > 0)
                vis = inb & (zc <= sd[j, vi, ui, 0] + tol)
                grid = torch.stack([2 * uv[:, 0] / Ws - 1, 2 * uv[:, 1] / Hs - 1], 1)[None, None]
                src = images[i].permute(2, 0, 1)[None].float() / 255
                col = F.grid_sample(src, grid, mode="bilinear", align_corners=False)[0, :, 0].T
                M, b = self.to_canon[c]
                col = (col - b) @ M.T
                cos = (dv * F.normalize(centers[i] - Xw, dim=1)).sum(1)
                w = vis.float() * (cos + 1.01) ** 8
                rgb += col * w[:, None]
                wsum += w
                fallback += col * inb.float()[:, None]
            people = torch.where((wsum > 0)[:, None], rgb / wsum.clamp(min=1e-6)[:, None],
                                 fallback / len(chosen))
            img = img.clone()
            img[ys, xs] += wp[ys, xs, None] * people.clamp(0, 1)
        return img.clamp(0, 1)

    @torch.no_grad()
    def render_view(self, state: dict, camera: Camera, timer: StageTimer, as_tensor: bool = False):
        """RGB uint8 (H, W, 3): numpy, or a GPU tensor with as_tensor (e.g. for GPU JPEG encoding)."""
        if self.texture == "pixel":
            out = (self._render_pixel(state, camera, timer) * 255).byte()
            if as_tensor:
                return out
            with timer("to_cpu"):
                return out.cpu().numpy()
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
                torch.cat([torch.cat([((rgb - 0.5) / SH_C0)[:, None],
                                      torch.zeros(n, bg["shN"].shape[1], 3, device=dev)], 1),
                           torch.cat([bg["sh0"], bg["shN"]], 1)]),
                self.bg.viewmat(camera)[None], K, camera.width, camera.height, sh_degree=self.bg.sh_degree,
                packed=False)
        out = (img[0].clamp(0, 1) * 255).byte()
        if as_tensor:
            return out
        with timer("to_cpu"):
            return out.cpu().numpy()

    # ---- offline-protocol interface (same held-out evaluation as every other method) ----
    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        if frame not in self.frame_cache:
            self.frame_cache = {frame: self.process(
                {c: self.seq.image(c, frame) for c in self.seq.valid_cameras(frame, self.train_cams)}, timer)}
        return self.render_view(self.frame_cache[frame], camera, timer)
