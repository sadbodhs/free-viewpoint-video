"""Simulated live camera streams, frame synchronization and per-frame health checks.

Discrete-event replay: every camera captures at the sequence frame rate; each packet arrives
after latency + jitter (+ faults). The synchronizer assembles one frame set per 1/fps slot
from packets that arrived before the slot deadline, so late/missing cameras are simply
invalid for that slot -- the pipeline never waits.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from fvv.data import MultiViewSequence
from .faults import FaultConfig


@dataclass
class Packet:
    cam: str
    frame: int               # source frame id (capture index)
    capture_t: float         # seconds
    arrival_t: float         # seconds
    image: torch.Tensor | None  # RGB uint8 (H, W, 3) on the GPU, as decoded (may be degraded / resized)
    decode_error: bool = False


@dataclass
class FrameSet:
    slot: int
    t: float
    frame: int
    images: dict[str, torch.Tensor] = field(default_factory=dict)  # valid cameras only (GPU)
    status: dict[str, str] = field(default_factory=dict)          # cam -> ok|missing|late|frozen|blank|blur|corrupt|resized


class ReplayStreams:
    """Replays a MultiViewSequence as live streams with faults. Frames are preloaded into GPU
    memory, as an NVDEC ingest would deliver them (decode itself is out of the measured path)."""

    def __init__(self, seq: MultiViewSequence, cams: list[str], frames: list[int], faults: FaultConfig,
                 fps: float = 30.0, device: str = "cuda"):
        self.seq, self.cams, self.frames, self.faults, self.fps = seq, cams, frames, faults, fps
        self.rng = np.random.default_rng(faults.seed)
        # includes the dataset's own dropped (blank) frames, so health checks see them live
        self.cache = {(c, f): torch.from_numpy(seq.image(c, f)).to(device) for f in frames for c in cams}

    def packets(self, k: int) -> list[Packet]:
        """All packets captured at slot k (one per camera, unless dropped)."""
        f, t = self.frames[k], k / self.fps
        fc = self.faults
        out = []
        for c in self.cams:
            if any(w.active(c, t) for w in fc.drops):
                continue
            src = f
            if any(w.active(c, t) for w in fc.freezes):
                start = min(w.start for w in fc.freezes if w.active(c, t))
                src = self.frames[min(int(start * self.fps), k)]
            img = self.cache.get((c, src))
            arrival = t + (fc.latency_ms + abs(self.rng.normal(0, fc.jitter_ms))) / 1000
            err = False
            if img is not None:
                if any(w.active(c, t) for w in fc.blurs):
                    img = _chw_to_hwc(TF.gaussian_blur(_hwc_to_chw(img), 37, 6.0))
                for cam, tc in fc.corrupts:          # invalid from error until next keyframe
                    if cam == c and tc <= t < (int(tc * self.fps) // fc.gop + 1) * fc.gop / self.fps:
                        err = True
                for cam, tr, sc in fc.res_changes:
                    if cam == c and t >= tr:
                        H, W = img.shape[:2]
                        img = _resize(img, (int(H * sc), int(W * sc)), "area")
            out.append(Packet(c, src, t, arrival, img, err))
        return out


def _hwc_to_chw(img: torch.Tensor) -> torch.Tensor:
    return img.permute(2, 0, 1).float()


def _chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 255).round().byte().permute(1, 2, 0).contiguous()


def _resize(img: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    kw = {} if mode == "area" else {"align_corners": False}
    return _chw_to_hwc(F.interpolate(_hwc_to_chw(img)[None], size=size, mode=mode, **kw)[0])


class Synchronizer:
    """Builds frame sets on a fixed clock; a packet counts for slot k if it was captured in that
    slot and arrived before slot_time + buffer."""

    def __init__(self, cams: list[str], expected_size: dict[str, tuple[int, int]], fps: float = 30.0,
                 buffer_ms: float = 100.0):
        self.cams, self.expected, self.fps, self.buffer = cams, expected_size, fps, buffer_ms / 1000
        self.last_small: dict[str, tuple[int, torch.Tensor]] = {}   # cam -> (slot, thumbnail)
        self.sharp_ref: dict[str, float] = {}
        self.lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)[None, None]

    def assemble(self, k: int, frame: int, packets: list[Packet]) -> FrameSet:
        t = k / self.fps
        fs = FrameSet(k, t, frame, status={c: "missing" for c in self.cams})
        cand, imgs = [], []
        for p in packets:
            if p.arrival_t > t + self.buffer:
                fs.status[p.cam] = "late"
                continue
            if p.decode_error or p.image is None:
                fs.status[p.cam] = "corrupt" if p.decode_error else "missing"
                continue
            img, W, H = p.image, *self.expected[p.cam]
            if img.shape[1] != W or img.shape[0] != H:
                # resolution changed mid-stream: same field of view assumed -> resample to calibrated size
                img = _resize(img, (H, W), "bilinear")
                fs.status[p.cam] = "resized"
            else:
                fs.status[p.cam] = "ok"
            cand.append(p.cam)
            imgs.append(img)
        if cand:
            for c, img, bad in zip(cand, imgs, self._health_batch(cand, imgs, k)):
                if bad:
                    fs.status[c] = bad
                else:
                    fs.images[c] = img
        return fs

    def _health_batch(self, cams: list[str], imgs: list[torch.Tensor], k: int) -> list[str | None]:
        """Blank / frozen / blur checks for all cameras at once on 1/8-res thumbnails, with a single
        GPU->CPU transfer per frame (per-camera .item() calls were most of the old cost)."""
        x = torch.stack(imgs).permute(0, 3, 1, 2).half()                    # (N, 3, H, W)
        thumbs = F.avg_pool2d(x, 8)                                          # (N, 3, H/8, W/8)
        std = thumbs.flatten(2).float().std(2).amax(1)
        gray = thumbs.float().mean(1, keepdim=True)
        sharp = F.conv2d(gray, self.lap.to(gray.device)).flatten(1).var(1)
        # frozen = identical thumbnail for a *new* slot (re-reading the same slot, e.g. paused, is fine)
        frozen = torch.zeros(len(cams), dtype=torch.bool, device=x.device)
        for i, c in enumerate(cams):
            prev = self.last_small.get(c)
            if prev is not None and prev[0] != k and prev[1].shape == thumbs[i].shape:
                frozen[i] = (prev[1] == thumbs[i]).all()          # stays on GPU (no sync)
            self.last_small[c] = (k, thumbs[i])
        stats = torch.stack([std, sharp, frozen.float()], 1).cpu().numpy()  # the one sync
        out = []
        for c, (sd, sh, fr) in zip(cams, stats):
            if sd < 8:
                out.append("blank")
                continue
            if fr:
                out.append("frozen")
                continue
            ref = self.sharp_ref.setdefault(c, float(sh))
            self.sharp_ref[c] = 0.95 * ref + 0.05 * float(sh) if sh > 0.5 * ref else ref
            out.append("blur" if sh < 0.3 * ref else None)
        return out
