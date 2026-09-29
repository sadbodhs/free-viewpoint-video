"""Fault injection for live-replay: what real RTSP camera feeds do to a pipeline.

Spec strings (comma-separated, times in seconds from the start of the replay):
    jitter=40                    network jitter std (ms), all cameras
    latency=60                   base network latency (ms)
    drop=00_05@1.0-2.5           camera outage (no packets)
    freeze=00_07@2.0-3.0         stream repeats its last frame
    blur=00_03@0.5-4.0           heavy defocus/motion blur
    corrupt=00_10@1.5            decode error at t; invalid until next keyframe (gop frames)
    res=00_04@2.0x0.5            resolution change at t (new scale factor)
    gop=30                       keyframe interval in frames
    seed=0
"""
from dataclasses import dataclass, field


@dataclass
class Window:
    cam: str
    start: float
    end: float

    def active(self, cam: str, t: float) -> bool:
        return cam == self.cam and self.start <= t < self.end


@dataclass
class FaultConfig:
    jitter_ms: float = 0.0
    latency_ms: float = 30.0
    gop: int = 30
    seed: int = 0
    drops: list[Window] = field(default_factory=list)
    freezes: list[Window] = field(default_factory=list)
    blurs: list[Window] = field(default_factory=list)
    corrupts: list[tuple[str, float]] = field(default_factory=list)
    res_changes: list[tuple[str, float, float]] = field(default_factory=list)

    @classmethod
    def parse(cls, spec: str | None) -> "FaultConfig":
        cfg = cls()
        for item in filter(None, (spec or "").split(",")):
            key, val = item.split("=", 1)
            if key in ("drop", "freeze", "blur"):
                cam, span = val.split("@")
                a, b = map(float, span.split("-"))
                {"drop": cfg.drops, "freeze": cfg.freezes, "blur": cfg.blurs}[key].append(Window(cam, a, b))
            elif key == "corrupt":
                cam, t = val.split("@")
                cfg.corrupts.append((cam, float(t)))
            elif key == "res":
                cam, rest = val.split("@")
                t, s = rest.split("x")
                cfg.res_changes.append((cam, float(t), float(s)))
            elif key == "jitter":
                cfg.jitter_ms = float(val)
            elif key == "latency":
                cfg.latency_ms = float(val)
            elif key == "gop":
                cfg.gop = int(val)
            elif key == "seed":
                cfg.seed = int(val)
            else:
                raise ValueError(f"unknown fault {key!r}")
        return cfg
