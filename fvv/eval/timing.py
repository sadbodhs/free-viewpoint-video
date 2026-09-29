"""Per-stage latency profiling (CUDA-synchronized so GPU work is counted)."""
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


class StageTimer:
    def __init__(self):
        self.times: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def __call__(self, stage: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times[stage].append((time.perf_counter() - t0) * 1000)

    def summary(self) -> dict[str, dict[str, float]]:
        return {k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95)), "n": len(v)}
                for k, v in self.times.items()}
