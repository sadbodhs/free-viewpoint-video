"""Simulated live camera streams, frame synchronization and per-frame health checks.

Discrete-event replay: every camera captures at the sequence frame rate; each packet arrives
after latency + jitter (+ faults). The synchronizer assembles one frame set per 1/fps slot
from packets that arrived before the slot deadline, so late/missing cameras are simply
invalid for that slot -- the pipeline never waits.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from fvv.data import MultiViewSequence
from .faults import FaultConfig


@dataclass
class Packet:
    cam: str
    frame: int               # source frame id (capture index)
    capture_t: float         # seconds
    arrival_t: float         # seconds
    image: np.ndarray | None # RGB uint8 as decoded (may be degraded / resized)
    decode_error: bool = False


@dataclass
class FrameSet:
    slot: int
    t: float
    frame: int
    images: dict[str, np.ndarray] = field(default_factory=dict)   # valid cameras only
    status: dict[str, str] = field(default_factory=dict)          # cam -> ok|missing|late|frozen|blank|blur|corrupt|resized


class ReplayStreams:
    """Replays a MultiViewSequence as live streams with faults. Frames are preloaded (decode is
    done by NVDEC in production; here it is out of the measured path)."""

    def __init__(self, seq: MultiViewSequence, cams: list[str], frames: list[int], faults: FaultConfig,
                 fps: float = 30.0):
        self.seq, self.cams, self.frames, self.faults, self.fps = seq, cams, frames, faults, fps
        self.rng = np.random.default_rng(faults.seed)
        # includes the dataset's own dropped (blank) frames, so health checks see them live
        self.cache = {(c, f): seq.image(c, f) for f in frames for c in cams}

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
                    img = cv2.GaussianBlur(img, (0, 0), 6)
                for cam, tc in fc.corrupts:          # invalid from error until next keyframe
                    if cam == c and tc <= t < (int(tc * self.fps) // fc.gop + 1) * fc.gop / self.fps:
                        err = True
                for cam, tr, s in fc.res_changes:
                    if cam == c and t >= tr:
                        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            out.append(Packet(c, src, t, arrival, img, err))
        return out


class Synchronizer:
    """Builds frame sets on a fixed clock; a packet counts for slot k if it was captured in that
    slot and arrived before slot_time + buffer."""

    def __init__(self, cams: list[str], expected_size: dict[str, tuple[int, int]], fps: float = 30.0,
                 buffer_ms: float = 100.0):
        self.cams, self.expected, self.fps, self.buffer = cams, expected_size, fps, buffer_ms / 1000
        self.last_hash: dict[str, float] = {}
        self.sharp_ref: dict[str, float] = {}

    def assemble(self, k: int, frame: int, packets: list[Packet]) -> FrameSet:
        t = k / self.fps
        fs = FrameSet(k, t, frame, status={c: "missing" for c in self.cams})
        for p in packets:
            if p.arrival_t > t + self.buffer:
                fs.status[p.cam] = "late"
                continue
            if p.decode_error or p.image is None:
                fs.status[p.cam] = "corrupt" if p.decode_error else "missing"
                continue
            fs.status[p.cam], img = self._health(p)
            if img is not None:
                fs.images[p.cam] = img
        return fs

    def _health(self, p: Packet) -> tuple[str, np.ndarray | None]:
        img = p.image
        W, H = self.expected[p.cam]
        status = "ok"
        if img.shape[1] != W or img.shape[0] != H:
            # resolution changed mid-stream: same field of view assumed -> resample to calibrated size
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            status = "resized"
        small = cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        if small.reshape(-1, 3).std(0).max() < 8:
            return "blank", None
        h = float(small.astype(np.float32).mean() + small[::7, ::7].astype(np.float32).std())
        if self.last_hash.get(p.cam) == h:
            return "frozen", None
        self.last_hash[p.cam] = h
        sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var()
        ref = self.sharp_ref.setdefault(p.cam, sharp)
        self.sharp_ref[p.cam] = 0.95 * ref + 0.05 * sharp if sharp > 0.5 * ref else ref
        if sharp < 0.3 * ref:
            return "blur", None
        return status, img
