"""Layered dynamic Gaussian splatting: static background + tracked people.

1. Background: one 3DGS model trained on key frames spread over the clip, with the people
   masked out of the loss. People move, so every part of the floor/walls is seen in some frame.
2. People (frame 0): Gaussians initialized inside skeleton capsules and optimized with the
   background frozen (MCMC densification restricted to the people).
3. Tracking (every next frame):
   a. warm start: move each Gaussian with its nearest bone's rigid motion (skeleton t-1 -> t);
   b. relocate dead (transparent) Gaussians onto live ones, so newly visible body parts and
      people entering the scene get Gaussians;
   c. optimize all people parameters with a local-rigidity loss; after every step Gaussians are
      projected back inside the capsules and size-capped (hard people prior, no streaks).
      The background never changes: no flicker, no ghosts.
The background + per-frame people split is also the layout of the real-time system (docs/REALTIME.md).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy
from gsplat.strategy.ops import relocate, sample_add

from fvv.data import Camera, MultiViewSequence
from fvv.data.people import bone_motion, bones, capsule_mask, sample_capsules
from fvv.eval import StageTimer
from fvv.geometry import load_points_ply
from .gaussian_splat import SH_C0, GaussianSplat, apply_affine, photometric_loss

FG_KEYS = ("means", "scales", "quats", "opacities", "sh0")


def seg_distance(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, chunk: int = 100_000):
    """Min distance from points x (N,3) to segments a->b (S,3), and the nearest segment index."""
    ab = b - a
    denom = (ab * ab).sum(1).clamp(min=1e-12)
    dist, idx = [], []
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk, None]
        t = (((xi - a) * ab).sum(-1) / denom).clamp(0, 1)
        d = (xi - (a + t[..., None] * ab)).norm(dim=-1)
        m = d.min(1)
        dist.append(m.values)
        idx.append(m.indices)
    return torch.cat(dist), torch.cat(idx)


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product, wxyz."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


class LayeredSplat:
    name = "layered"
    single_frame = False

    def __init__(self, bg_frames: int = 12, bg_iters: int = 30_000, fg_iters: int = 5_000,
                 steps_per_frame: int = 500, radius: float = 35.0, fg_cap: int = 250_000,
                 rigidity: float = 1e3, max_scale: float = 6.0, knn: int = 8, seed: int = 0,
                 ckpt_dir: str | Path | None = None, retrain: bool = False, device: str = "cuda", **_):
        self.bg_frames, self.bg_iters, self.fg_iters = bg_frames, bg_iters, fg_iters
        self.steps, self.radius, self.fg_cap = steps_per_frame, radius, fg_cap
        self.rigidity, self.max_scale, self.knn, self.seed = rigidity, max_scale, knn, seed
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.retrain, self.device = retrain, device
        self.bg = GaussianSplat(iters=bg_iters, seed=seed, device=device)
        self.bg.name = "layered_bg"
        self.states: dict[int, dict[str, torch.Tensor]] = {}

    # ---- helpers -----------------------------------------------------------------------
    def _norm(self, X: np.ndarray) -> np.ndarray:
        return (X - self.bg.center) / self.bg.scale

    def _bones_t(self, seq: MultiViewSequence, frame: int):
        """(keys, a, b) of observed bones at `frame`, as normalized-coordinate tensors."""
        bs = bones(seq.bodies(frame))
        keys = list(bs)
        a = torch.tensor(self._norm(np.array([bs[k][0] for k in keys])), dtype=torch.float32, device=self.device)
        b = torch.tensor(self._norm(np.array([bs[k][1] for k in keys])), dtype=torch.float32, device=self.device)
        return keys, a, b, bs

    def _views(self, seq: MultiViewSequence, frame: int, cams: list[str], with_masks: bool = False):
        cams = [seq.camera(c) for c in seq.valid_cameras(frame, cams)]
        images = torch.stack([torch.from_numpy(seq.image(c.name, frame)) for c in cams]).to(self.device)
        viewmats = torch.stack([self.bg.viewmat(c) for c in cams])
        Ks = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32, device=self.device)
        ids = torch.tensor([self.cam_index[c.name] for c in cams], device=self.device)
        masks = None
        if with_masks:
            segs = list(bones(seq.bodies(frame)).values())
            masks = torch.from_numpy(np.stack([~capsule_mask(c, segs, self.radius) for c in cams])).to(self.device)
        return images, viewmats, Ks, ids, masks, cams[0].width, cams[0].height

    def _render(self, fg: dict, viewmats, Ks, W, H):
        """Rasterize people (fg, degree-0 color) together with the frozen background."""
        bg = self.bg.params
        n = len(fg["means"])
        return rasterization(
            torch.cat([fg["means"], bg["means"]]), torch.cat([fg["quats"], bg["quats"]]),
            torch.exp(torch.cat([fg["scales"], bg["scales"]])),
            torch.sigmoid(torch.cat([fg["opacities"], bg["opacities"]])),
            torch.cat([torch.cat([fg["sh0"], torch.zeros(n, 15, 3, device=self.device)], 1),
                       torch.cat([bg["sh0"], bg["shN"]], 1)]),
            viewmats, Ks, W, H, sh_degree=3, packed=False, near_plane=0.01)

    # ---- fit ---------------------------------------------------------------------------
    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        track = seq.frame_ids
        self.train_cams = train_cams
        self.cam_index = {c: k for k, c in enumerate(train_cams)}
        bg_ckpt = self.ckpt_dir / f"layered_bg_{seq.root.name}.pt" if self.ckpt_dir else None
        fg_ckpt = self.ckpt_dir / f"layered_fg_{seq.root.name}_{track[0]:08d}_{track[-1]:08d}.pt" if self.ckpt_dir else None

        if bg_ckpt and bg_ckpt.exists() and not self.retrain:
            self.bg.load(bg_ckpt)
            print(f"[layered] loaded background {bg_ckpt}")
        else:
            self._fit_background(seq, track)
            if bg_ckpt:
                self.bg.save(bg_ckpt)
        for p in self.bg.params.values():
            p.requires_grad_(False)

        if fg_ckpt and fg_ckpt.exists() and not self.retrain:
            self.states = torch.load(fg_ckpt, map_location=self.device, weights_only=False)
            print(f"[layered] loaded {len(self.states)} people states {fg_ckpt}")
            return
        fg = self._fit_people_frame0(seq, track[0])
        self.states = {track[0]: {k: v.detach().half().clone() for k, v in fg.items()}}
        t0 = time.time()
        for k in range(1, len(track)):
            fg, loss = self._track(seq, fg, track[k - 1], track[k])
            self.states[track[k]] = {kk: v.detach().half().clone() for kk, v in fg.items()}
            if k % 10 == 0 or k == len(track) - 1:
                print(f"[layered] frame {track[k]} ({k}/{len(track) - 1})  loss {loss:.4f}  "
                      f"people gaussians {len(fg['means']):,}  {time.time() - t0:5.0f}s", flush=True)
        if fg_ckpt:
            torch.save(self.states, fg_ckpt)
            print(f"[layered] saved {fg_ckpt}")

    def _fit_background(self, seq: MultiViewSequence, track: list[int]) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cams0 = [seq.camera(c) for c in self.train_cams]
        C = np.stack([c.center for c in cams0])
        self.bg.center = C.mean(0)
        self.bg.scale = float(np.linalg.norm(C - self.bg.center, axis=1).max())
        self.bg.frame = track[0]
        self.bg.cam_names = list(self.train_cams)

        keyframes = track[:: max(1, len(track) // self.bg_frames)][: self.bg_frames]
        imgs, vms, Ks, ids, masks = [], [], [], [], []
        for f in keyframes:
            im, vm, K, i, m, W, H = self._views(seq, f, self.train_cams, with_masks=True)
            imgs.append(im), vms.append(vm), Ks.append(K), ids.append(i), masks.append(m)
        imgs, vms, Ks, ids, masks = map(torch.cat, (imgs, vms, Ks, ids, masks))
        print(f"[layered] background: {len(keyframes)} key frames, {len(imgs)} images, "
              f"{(~masks).float().mean().item():.1%} of pixels masked as people")

        # init: triangulated points minus those on people, plus random visible points
        vm0 = torch.stack([self.bg.viewmat(c) for c in cams0])
        K0 = torch.tensor(np.stack([c.K for c in cams0]), dtype=torch.float32, device=self.device)
        pts_path = seq.root / "points" / f"{track[0]:08d}.ply"
        if pts_path.exists():
            pts, cols = load_points_ply(pts_path)
            _, a, b, _ = self._bones_t(seq, track[0])
            d, _ = seg_distance(torch.tensor(self._norm(pts), dtype=torch.float32, device=self.device), a, b)
            keep = (d * self.bg.scale > self.radius).cpu().numpy()
            self.bg._init_gaussians(vm0, K0, W, H, points=self._norm(pts[keep]), colors=cols[keep], random_frac=0.5)
        else:
            self.bg._init_gaussians(vm0, K0, W, H)
        self.bg._train(imgs, vms, Ks, W, H, cam_ids=ids, masks=masks)

    def _new_fg(self, x: torch.Tensor, rgb: torch.Tensor) -> dict:
        n = len(x)
        knn = torch.cat([torch.cdist(x[i:i + 4096], x).topk(4, largest=False).values[:, 1:].mean(1)
                         for i in range(0, n, 4096)])
        return {
            "means": torch.nn.Parameter(x),
            "scales": torch.nn.Parameter(torch.log(knn.clamp(min=1e-4))[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=self.device), dim=1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=self.device))),
            "sh0": torch.nn.Parameter(((rgb - 0.5) / SH_C0)[:, None]),
        }

    def _optimize(self, fg: dict, seq, frame, steps, lrs, strategy=None, prev_means=None, nbrs=None):
        """Optimize people Gaussians on one frame's training views (background frozen)."""
        images, viewmats, Ks, ids, _, W, H = self._views(seq, frame, self.train_cams)
        _, a, b, _ = self._bones_t(seq, frame)
        opts = {k: torch.optim.Adam([{"params": fg[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in FG_KEYS}
        state = strategy.initialize_state() if strategy else None
        prev_off = None if prev_means is None else prev_means[:, None] - prev_means[nbrs]
        A = self.bg.affine
        for step in range(steps):
            v = step % len(images)
            render, _, info = self._render(fg, viewmats[v:v + 1], Ks[v:v + 1], W, H)
            pred = apply_affine(render[0], A[ids[v]]).clamp(0, 1)
            loss = photometric_loss(pred, images[v].float() / 255)
            if prev_off is not None:
                off = fg["means"][:, None] - fg["means"][nbrs]
                loss = loss + self.rigidity * ((off - prev_off) ** 2).sum(-1).mean()
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            if strategy:
                strategy.step_post_backward(fg, opts, state, step, info, lr=lrs["means"])
            self._constrain(fg, a, b)
        return loss.item()

    @torch.no_grad()
    def _constrain(self, fg: dict, a: torch.Tensor, b: torch.Tensor) -> None:
        """Hard people prior: project Gaussians that left the capsules back onto their surface,
        and cap their size so they cannot stretch into streaks across the background."""
        r = self.radius / self.bg.scale
        x = fg["means"].data
        d, idx = seg_distance(x, a, b)
        out = d > r
        if out.any():
            ai, bi, xo = a[idx[out]], b[idx[out]], x[out]
            ab = bi - ai
            t = (((xo - ai) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0, 1)
            c = ai + t[:, None] * ab
            x[out] = c + (xo - c) * (r / d[out])[:, None]
        fg["scales"].data.clamp_(max=float(np.log(self.max_scale / self.bg.scale)))

    def _fit_people_frame0(self, seq: MultiViewSequence, frame: int) -> dict:
        rng = np.random.default_rng(self.seed)
        segs = list(bones(seq.bodies(frame)).values())
        x = torch.tensor(self._norm(sample_capsules(segs, self.radius, 50_000, rng)), dtype=torch.float32,
                         device=self.device)
        fg = self._new_fg(x, torch.full((len(x), 3), 0.5, device=self.device))
        strategy = MCMCStrategy(cap_max=self.fg_cap, noise_lr=5e5, refine_start_iter=100,
                                refine_stop_iter=self.fg_iters - 500, refine_every=100)
        lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
        loss = self._optimize(fg, seq, frame, self.fg_iters, lrs, strategy=strategy)
        print(f"[layered] people at frame {frame}: {len(fg['means']):,} gaussians, loss {loss:.4f}")
        return fg

    @torch.no_grad()
    def _warm_start(self, fg: dict, seq, f_prev: int, f_cur: int) -> None:
        """Move each Gaussian rigidly with its nearest bone (skeleton f_prev -> f_cur)."""
        keys, a, b, bs_prev = self._bones_t(seq, f_prev)
        _, nearest = seg_distance(fg["means"], a, b)
        motion = bone_motion(bs_prev, bones(seq.bodies(f_cur)))
        R = np.tile(np.eye(3), (len(keys), 1, 1))
        a0, a1 = np.zeros((len(keys), 3)), np.zeros((len(keys), 3))
        for i, k in enumerate(keys):
            if k in motion:
                R[i], a0[i], a1[i] = motion[k][0], self._norm(motion[k][1]), self._norm(motion[k][2])
            else:  # bone lost this frame: keep Gaussians in place
                a0[i] = a1[i] = self._norm(bs_prev[k][0])
        Rt = torch.tensor(R, dtype=torch.float32, device=self.device)[nearest]
        a0t = torch.tensor(a0, dtype=torch.float32, device=self.device)[nearest]
        a1t = torch.tensor(a1, dtype=torch.float32, device=self.device)[nearest]
        fg["means"].data = torch.einsum("nij,nj->ni", Rt, fg["means"] - a0t) + a1t
        q = torch.tensor(np.stack([rotmat_to_quat(r) for r in R]), dtype=torch.float32, device=self.device)[nearest]
        fg["quats"].data = quat_mul(q, F.normalize(fg["quats"], dim=1))

    def _track(self, seq, fg: dict, f_prev: int, f_cur: int):
        fg = {k: torch.nn.Parameter(v.detach().clone()) for k, v in fg.items()}
        self._warm_start(fg, seq, f_prev, f_cur)
        # relocate transparent Gaussians onto visible ones; grow a little (up to the cap)
        opts = {k: torch.optim.Adam([{"params": fg[k], "lr": 0.0, "name": k}]) for k in FG_KEYS}
        binoms = MCMCStrategy().initialize_state()["binoms"].to(self.device)
        dead = torch.sigmoid(fg["opacities"]) <= 0.005
        if dead.any():
            relocate(fg, opts, {}, dead, binoms, min_opacity=0.005)
        n_add = min(self.fg_cap, int(len(fg["means"]) * 1.02)) - len(fg["means"])
        if n_add > 0:
            sample_add(fg, opts, {}, n_add, binoms, min_opacity=0.005)
        x = fg["means"].detach()
        nbrs = torch.cat([torch.cdist(x[i:i + 2048], x).topk(self.knn + 1, largest=False).indices[:, 1:]
                          for i in range(0, len(x), 2048)])
        lrs = {"means": 2e-4, "scales": 2e-3, "quats": 1e-3, "opacities": 1e-2, "sh0": 1e-3}
        loss = self._optimize(fg, seq, f_cur, self.steps, lrs, prev_means=x.clone(), nbrs=nbrs)
        return fg, loss

    # ---- render ------------------------------------------------------------------------
    @torch.no_grad()
    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        with timer("load_state"):
            fg = {k: v.float() for k, v in self.states[frame].items()}
        with timer("rasterize"):
            K = torch.tensor(camera.K, dtype=torch.float32, device=self.device)[None]
            img, _, _ = self._render(fg, self.bg.viewmat(camera)[None], K, camera.width, camera.height)
        with timer("to_cpu"):
            return (img[0].clamp(0, 1) * 255).byte().cpu().numpy()
