"""Static 3D Gaussian Splatting on a single frame (gsplat rasterizer + MCMC densification).

Phase 1 quality reference: optimize one frame from all training cameras, render any viewpoint.
Gaussians start at points triangulated from the training cameras (scripts/triangulate_points.py);
random init is a fallback, but in an inward-facing rig it leaves view-dependent floaters.
The world is normalized so camera centers lie within a unit sphere (fixed learning rates work
for any rig size or unit).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

from fvv.data import Camera, MultiViewSequence
from fvv.eval import StageTimer
from fvv.geometry import load_points_ply

SH_C0 = 0.28209479177387814


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of (B, 3, H, W) images in [0, 1]."""
    r = torch.arange(window, device=x.device, dtype=x.dtype) - window // 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()
    conv = lambda t: F.conv2d(t, w, padding=window // 2, groups=3)
    mx, my = conv(x), conv(y)
    sxx, syy, sxy = conv(x * x) - mx ** 2, conv(y * y) - my ** 2, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


class GaussianSplat:
    name = "3dgs"
    single_frame = True

    def __init__(self, iters: int = 30_000, cap_max: int = 1_000_000, init_points: int = 300_000,
                 color_affine: bool = True, seed: int = 0, ckpt_dir: str | Path | None = None,
                 retrain: bool = False, device: str = "cuda"):
        self.iters, self.cap_max, self.init_points = iters, cap_max, init_points
        self.color_affine, self.seed = color_affine, seed
        self.name = ("3dgs" if color_affine else "3dgs_noaffine") + (f"_s{seed}" if seed else "")
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.retrain, self.device = retrain, device
        self.params: torch.nn.ParameterDict | None = None

    # ---- coordinates -------------------------------------------------------------------
    def viewmat(self, cam: Camera) -> torch.Tensor:
        """World->camera in normalized coordinates (X_n = (X - center) / scale)."""
        T = np.eye(4)
        T[:3, :3] = cam.R
        T[:3, 3] = (cam.R @ self.center + cam.t) / self.scale
        return torch.tensor(T, dtype=torch.float32, device=self.device)

    def _ckpt_path(self, seq: MultiViewSequence, frame: int) -> Path | None:
        return self.ckpt_dir / f"{self.name}_{seq.root.name}_{frame:08d}.pt" if self.ckpt_dir else None

    # ---- fit ---------------------------------------------------------------------------
    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        self.frame = frames[0]
        ckpt = self._ckpt_path(seq, self.frame)
        if ckpt and ckpt.exists() and not self.retrain:
            self.load(ckpt)
            print(f"[3dgs] loaded {ckpt} ({len(self.params['means'])} gaussians)")
            return

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cams = [seq.camera(c) for c in seq.valid_cameras(self.frame, train_cams)]
        C = np.stack([c.center for c in cams])
        self.center = C.mean(0)
        self.scale = float(np.linalg.norm(C - self.center, axis=1).max())

        W, H = cams[0].width, cams[0].height
        assert all((c.width, c.height) == (W, H) for c in cams), "mixed resolutions not supported yet"
        images = torch.stack([torch.from_numpy(seq.image(c.name, self.frame)) for c in cams])
        images = images.to(self.device).float() / 255.0                         # (V, H, W, 3)
        viewmats = torch.stack([self.viewmat(c) for c in cams])
        Ks = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32, device=self.device)

        pts_path = seq.root / "points" / f"{self.frame:08d}.ply"
        if pts_path.exists():
            pts, cols = load_points_ply(pts_path)
            print(f"[3dgs] init from {len(pts):,} triangulated points ({pts_path})")
            self._init_gaussians(viewmats, Ks, W, H, points=(pts - self.center) / self.scale, colors=cols)
        else:
            print("[3dgs] no triangulated points (scripts/triangulate_points.py); random init")
            self._init_gaussians(viewmats, Ks, W, H)
        self._train(images, viewmats, Ks, W, H)
        if ckpt:
            self.save(ckpt)
            self.export_ply(ckpt.with_suffix(".ply"))
            print(f"[3dgs] saved {ckpt} and .ply")

    def _random_visible(self, n, viewmats, Ks, W, H, min_views: int = 3) -> torch.Tensor:
        """Uniform points in a sphere around the rig, kept if >= min_views cameras see them."""
        pts = []
        while sum(len(p) for p in pts) < n:
            x = torch.randn(n, 3, device=self.device)
            x = x / x.norm(dim=1, keepdim=True) * torch.rand(n, 1, device=self.device) ** (1 / 3) * 1.2
            xc = torch.einsum("vij,nj->vni", viewmats[:, :3, :3], x) + viewmats[:, None, :3, 3]
            uv = torch.einsum("vij,vnj->vni", Ks, xc)
            uv = uv[..., :2] / uv[..., 2:].clamp(min=1e-6)
            seen = (xc[..., 2] > 0.05) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
            pts.append(x[seen.sum(0) >= min_views])
        return torch.cat(pts)[:n]

    def _init_gaussians(self, viewmats, Ks, W, H, points=None, colors=None, random_frac: float = 0.1):
        """Gaussians at triangulated points (+ a few random ones for untextured regions), or all random."""
        dev = self.device
        if points is None:
            x = self._random_visible(self.init_points, viewmats, Ks, W, H)
            rgb = torch.rand(len(x), 3, device=dev)
            init_opacity = 0.5
        else:
            sfm = torch.tensor(points, dtype=torch.float32, device=dev)
            extra = self._random_visible(int(len(sfm) * random_frac), viewmats, Ks, W, H)
            x = torch.cat([sfm, extra])
            rgb = torch.cat([torch.tensor(colors, dtype=torch.float32, device=dev) / 255,
                             torch.full((len(extra), 3), 0.5, device=dev)])
            init_opacity = 0.1
        n = len(x)
        # scale = mean distance to 3 nearest neighbours (chunked to bound memory)
        knn = torch.cat([torch.cdist(x[i:i + 4096], x).topk(4, largest=False).values[:, 1:].mean(1)
                         for i in range(0, n, 4096)])
        self.params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(x),
            "scales": torch.nn.Parameter(torch.log(knn.clamp(min=1e-4))[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=dev), dim=1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opacity, device=dev))),
            "sh0": torch.nn.Parameter(((rgb - 0.5) / SH_C0)[:, None]),
            "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=dev)),
        })

    def _rasterize(self, viewmats, Ks, W, H, sh_degree=3):
        p = self.params
        return rasterization(
            p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            torch.cat([p["sh0"], p["shN"]], 1), viewmats, Ks, W, H,
            sh_degree=sh_degree, packed=False, near_plane=0.01)

    def _train(self, images, viewmats, Ks, W, H):
        lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
               "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        opts = {k: torch.optim.Adam([{"params": self.params[k], "lr": lr, "name": k}], eps=1e-15)
                for k, lr in lrs.items()}
        # means lr decays exponentially to 1% over training
        sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / self.iters))
        strategy = MCMCStrategy(cap_max=self.cap_max, noise_lr=5e5, refine_start_iter=500,
                                refine_stop_iter=int(self.iters * 0.83), refine_every=100)
        strategy.check_sanity(self.params, opts)
        state = strategy.initialize_state()

        # Per-training-camera affine color transform (identity init). Rig cameras differ in
        # exposure/white balance; without this, the Gaussians absorb those differences as
        # view-dependent floaters. Novel views render without it (shared color space).
        V = len(images)
        affine = torch.nn.Parameter(torch.eye(3, 4, device=self.device).repeat(V, 1, 1))
        affine_opt = torch.optim.Adam([affine], lr=5e-4) if self.color_affine else None

        t0 = time.time()
        order = torch.randperm(V)
        for step in range(self.iters):
            i = int(order[step % V])
            if step % V == V - 1:
                order = torch.randperm(V)
            render, _, info = self._rasterize(viewmats[i:i + 1], Ks[i:i + 1], W, H,
                                              sh_degree=min(step // 1000, 3))
            pred, gt = render[0], images[i]
            if affine_opt:
                # Gauge fix: transforms are relative to their mean, so the shared color space is
                # "the average camera" and cannot drift (e.g. darker scene + brighter transforms).
                A = affine[i] - affine.mean(0) + torch.eye(3, 4, device=self.device)
                pred = pred @ A[:, :3].T + A[:, 3]
            pred = pred.clamp(0, 1)
            l1 = (pred - gt).abs().mean()
            dssim = 1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            loss = (0.8 * l1 + 0.2 * dssim
                    + 0.01 * torch.sigmoid(self.params["opacities"]).mean()
                    + 0.01 * torch.exp(self.params["scales"]).mean())
            strategy.step_pre_backward(self.params, opts, state, step, info)
            loss.backward()
            for o in [*opts.values(), *([affine_opt] if affine_opt else [])]:
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            strategy.step_post_backward(self.params, opts, state, step, info, lr=sched.get_last_lr()[0])

            if step % 1000 == 0 or step == self.iters - 1:
                psnr = -10 * torch.log10(((pred - gt) ** 2).mean()).item()
                print(f"[3dgs] step {step:6d}  loss {loss.item():.4f}  train-psnr {psnr:5.2f}  "
                      f"gaussians {len(self.params['means']):,}  {time.time() - t0:6.0f}s", flush=True)

    # ---- render ------------------------------------------------------------------------
    @torch.no_grad()
    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        with timer("rasterize"):
            K = torch.tensor(camera.K, dtype=torch.float32, device=self.device)[None]
            img, _, _ = self._rasterize(self.viewmat(camera)[None], K, camera.width, camera.height)
        with timer("to_cpu"):
            return (img[0].clamp(0, 1) * 255).byte().cpu().numpy()

    # ---- io ----------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": {k: v.detach() for k, v in self.params.items()},
                    "center": self.center, "scale": self.scale, "frame": self.frame}, path)

    def load(self, path: Path) -> None:
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in d["params"].items()})
        self.center, self.scale, self.frame = d["center"], d["scale"], d["frame"]

    @torch.no_grad()
    def export_ply(self, path: Path) -> None:
        """Standard 3DGS .ply (as written by the original implementation), normalized coordinates."""
        p = {k: v.detach().cpu().numpy() for k, v in self.params.items()}
        n = len(p["means"])
        cols = {
            **{a: p["means"][:, i] for i, a in enumerate("xyz")},
            **{f"n{a}": np.zeros(n) for a in "xyz"},
            **{f"f_dc_{i}": p["sh0"][:, 0, i] for i in range(3)},
            **{f"f_rest_{i}": v for i, v in enumerate(p["shN"].transpose(0, 2, 1).reshape(n, -1).T)},
            "opacity": p["opacities"],
            **{f"scale_{i}": p["scales"][:, i] for i in range(3)},
            **{f"rot_{i}": v for i, v in enumerate((p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)).T)},
        }
        data = np.empty(n, dtype=[(k, "<f4") for k in cols])
        for k, v in cols.items():
            data[k] = v
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
                 "".join(f"property float {k}\n" for k in cols) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())
