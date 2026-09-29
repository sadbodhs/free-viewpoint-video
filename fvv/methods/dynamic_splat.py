"""Dynamic Gaussian splatting by per-frame tracking (after Luiten et al., "Dynamic 3D Gaussians").

1. Frame 0: a full static 3DGS model (GaussianSplat, reused from its checkpoint).
2. Split: Gaussians near people are dynamic; everything else is static and frozen for the
   whole sequence (no background flicker, no new floaters).
3. Every next frame: warm-start from the previous frame and optimize only the dynamic
   Gaussians' position/rotation (plus a little color/opacity) against that frame's images.
   A local-rigidity loss keeps neighbours moving together.

People are located with 3D skeletons (dataset-provided here; any multi-view 3D pose
estimator can supply them for other captures).
"""
import time
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

from fvv.data import COCO19_EDGES, Camera, MultiViewSequence
from fvv.eval import StageTimer
from .gaussian_splat import GaussianSplat, ssim


def _dist_to_segments(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
    """Min distance from points x (N,3) to segments a->b (S,3)."""
    out = []
    ab = b - a
    denom = (ab * ab).sum(1).clamp(min=1e-12)
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk, None]                                           # (n, 1, 3)
        t = (((xi - a) * ab).sum(-1) / denom).clamp(0, 1)                   # (n, S)
        out.append((xi - (a + t[..., None] * ab)).norm(dim=-1).min(1).values)
    return torch.cat(out)


class DynamicSplat:
    name = "dyn3dgs"
    single_frame = False

    def __init__(self, steps_per_frame: int = 500, person_radius: float = 30.0, rigidity: float = 1e3,
                 knn: int = 8, seed: int = 0, ckpt_dir: str | Path | None = None, retrain: bool = False,
                 device: str = "cuda", **base_kwargs):
        self.steps, self.person_radius, self.rigidity, self.knn = steps_per_frame, person_radius, rigidity, knn
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.retrain, self.device = retrain, device
        self.base = GaussianSplat(seed=seed, ckpt_dir=ckpt_dir, device=device, **base_kwargs)
        self.states: dict[int, dict[str, torch.Tensor]] = {}

    # ---- fit ---------------------------------------------------------------------------
    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        # always track the whole clip (evaluation only subsamples it), so one checkpoint serves all
        track = seq.frame_ids
        ckpt = self.ckpt_dir / f"{self.name}_{seq.root.name}_{track[0]:08d}_{track[-1]:08d}.pt" if self.ckpt_dir else None

        self.base.fit(seq, train_cams, [track[0]])
        if ckpt and ckpt.exists() and not self.retrain:
            d = torch.load(ckpt, map_location=self.device, weights_only=False)
            self._apply_permutation(d["perm"])
            self.n_dyn, self.states = d["n_dyn"], d["states"]
            print(f"[dyn] loaded {ckpt} ({len(self.states)} frames, {self.n_dyn:,} dynamic gaussians)")
            return

        self.train_cams = train_cams
        self._select_dynamic(seq, track[0])
        self._fit_affine(seq, track[0])
        self.states = {track[0]: self._dyn_state()}
        t0 = time.time()
        for k, f in enumerate(track[1:], 1):
            loss = self._track_frame(seq, f)
            self.states[f] = self._dyn_state()
            if k % 10 == 0 or k == len(track) - 1:
                print(f"[dyn] frame {f} ({k}/{len(track) - 1})  loss {loss:.4f}  {time.time() - t0:5.0f}s", flush=True)
        if ckpt:
            torch.save({"perm": self.perm, "n_dyn": self.n_dyn, "states": self.states}, ckpt)
            print(f"[dyn] saved {ckpt}")

    def _apply_permutation(self, perm: torch.Tensor) -> None:
        """Reorder base Gaussians so dynamic ones come first (params[:n_dyn])."""
        self.perm = perm
        for k in list(self.base.params.keys()):
            self.base.params[k] = torch.nn.Parameter(self.base.params[k].detach()[perm.to(self.device)])

    def _select_dynamic(self, seq: MultiViewSequence, frame: int) -> None:
        bodies = seq.bodies(frame)
        segs = [(b["joints"][i, :3], b["joints"][j, :3]) for b in bodies for i, j in COCO19_EDGES
                if b["joints"][i, 3] > 0 and b["joints"][j, 3] > 0]
        a = torch.tensor(np.array([s[0] for s in segs]), dtype=torch.float32, device=self.device)
        b = torch.tensor(np.array([s[1] for s in segs]), dtype=torch.float32, device=self.device)
        world = self.base.params["means"].detach() * self.base.scale + torch.tensor(
            self.base.center, dtype=torch.float32, device=self.device)
        dyn = _dist_to_segments(world, a, b) < self.person_radius
        self._apply_permutation(torch.cat([dyn.nonzero()[:, 0], (~dyn).nonzero()[:, 0]]).cpu())
        self.n_dyn = int(dyn.sum())
        # fixed neighbour graph among dynamic Gaussians for the rigidity loss
        x = self.base.params["means"].detach()[: self.n_dyn]
        self.nbrs = torch.cat([torch.cdist(x[i:i + 2048], x).topk(self.knn + 1, largest=False).indices[:, 1:]
                               for i in range(0, self.n_dyn, 2048)])
        print(f"[dyn] {self.n_dyn:,} dynamic / {len(dyn) - self.n_dyn:,} static gaussians "
              f"({len(bodies)} people, radius {self.person_radius} cm)")

    def _dyn_state(self) -> dict[str, torch.Tensor]:
        p, n = self.base.params, self.n_dyn
        return {k: p[k].detach()[:n].half().clone() for k in ("means", "quats", "sh0", "opacities")}

    def _load_state(self, frame: int) -> None:
        s = self.states[frame]
        for k, v in s.items():
            self.base.params[k].data[: self.n_dyn] = v.float()

    # ---- per-frame optimization --------------------------------------------------------
    def _frame_batch(self, seq: MultiViewSequence, frame: int):
        cams = [seq.camera(c) for c in seq.valid_cameras(frame, self.train_cams)]
        images = torch.stack([torch.from_numpy(seq.image(c.name, frame)) for c in cams]).to(self.device).float() / 255
        viewmats = torch.stack([self.base.viewmat(c) for c in cams])
        Ks = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32, device=self.device)
        idx = torch.tensor([self.cam_index[c.name] for c in cams], device=self.device)
        return images, viewmats, Ks, idx, cams[0].width, cams[0].height

    def _affine(self, i: torch.Tensor) -> torch.Tensor:
        return self.affine[i] - self.affine.mean(0) + torch.eye(3, 4, device=self.device)

    def _fit_affine(self, seq: MultiViewSequence, frame: int, steps: int = 300) -> None:
        """Per-camera color transforms for the frozen base model (same gauge as GaussianSplat)."""
        self.cam_index = {c: k for k, c in enumerate(self.train_cams)}
        self.affine = torch.nn.Parameter(torch.eye(3, 4, device=self.device).repeat(len(self.train_cams), 1, 1))
        opt = torch.optim.Adam([self.affine], lr=5e-3)
        images, viewmats, Ks, idx, W, H = self._frame_batch(seq, frame)
        with torch.no_grad():
            renders = torch.cat([self.base._rasterize(viewmats[v:v + 1], Ks[v:v + 1], W, H)[0]
                                 for v in range(len(images))])
        for _ in range(steps):
            A = self._affine(idx)
            pred = torch.einsum("vhwc,vdc->vhwd", renders, A[..., :3]) + A[:, None, None, :, 3]
            loss = (pred.clamp(0, 1) - images).abs().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.affine.requires_grad_(False)

    def _track_frame(self, seq: MultiViewSequence, frame: int) -> float:
        images, viewmats, Ks, idx, W, H = self._frame_batch(seq, frame)
        p, n = self.base.params, self.n_dyn
        prev = p["means"].detach()[:n].clone()
        dyn = {k: torch.nn.Parameter(p[k].detach()[:n].clone()) for k in ("means", "quats", "sh0", "opacities")}
        static = {k: p[k].detach()[n:] for k in p.keys()}
        scales, shN = torch.exp(p["scales"].detach()), p["shN"].detach()     # frozen for all Gaussians
        opt = torch.optim.Adam([
            {"params": [dyn["means"]], "lr": 2e-4}, {"params": [dyn["quats"]], "lr": 1e-3},
            {"params": [dyn["sh0"]], "lr": 5e-4}, {"params": [dyn["opacities"]], "lr": 5e-3}], eps=1e-15)
        prev_off = prev[:, None] - prev[self.nbrs]                            # neighbour offsets at t-1

        for step in range(self.steps):
            v = step % len(images)
            means = torch.cat([dyn["means"], static["means"]])
            render, _, _ = rasterization(
                means, torch.cat([dyn["quats"], static["quats"]]), scales,
                torch.sigmoid(torch.cat([dyn["opacities"], static["opacities"]])),
                torch.cat([torch.cat([dyn["sh0"], static["sh0"]]), shN], 1),
                viewmats[v:v + 1], Ks[v:v + 1], W, H, sh_degree=3, packed=False, near_plane=0.01)
            A = self._affine(idx[v])
            pred = (render[0] @ A[:, :3].T + A[:, 3]).clamp(0, 1)
            gt = images[v]
            off = dyn["means"][:, None] - dyn["means"][self.nbrs]
            loss = (0.8 * (pred - gt).abs().mean()
                    + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
                    + self.rigidity * ((off - prev_off) ** 2).sum(-1).mean())
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        for k, t in dyn.items():
            p[k].data[:n] = t.detach()
        return loss.item()

    # ---- render ------------------------------------------------------------------------
    @torch.no_grad()
    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        with timer("load_state"):
            self._load_state(frame)
        return self.base.render(frame, camera, timer)
