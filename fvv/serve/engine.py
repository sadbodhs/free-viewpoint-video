"""Live viewer engine: one GPU thread runs the live clock and renders every connected viewer.

Each tick (1/fps): advance the replayed "live" feeds, sync + health-check them, run the online
method (visual hull) once, then render the latest state from each viewer's own camera and
JPEG-encode it. Viewers only send camera poses; the heavy work is shared.
"""
import threading
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from fvv.data import Camera, MultiViewSequence
from fvv.eval import StageTimer
from fvv.live import FaultConfig, ReplayStreams, Synchronizer
from fvv.render import look_at, scene_frame

DEMO_FAULTS = "jitter=40,drop=00_05@1-3,freeze=00_07@2-3,blur=00_03@0.5-4,corrupt=00_10@1.5,res=00_04@2x0.5"


@dataclass
class Viewer:
    """Orbit camera around the scene target: azimuth/elevation (deg), distance (cm), zoom (x)."""
    az: float = 30.0
    el: float = 25.0
    dist: float = 250.0
    zoom: float = 1.0
    quality: int = 80
    follow: str = "group"                 # group | person | off
    target: np.ndarray | None = None      # current orbit center (world), smoothed
    person: np.ndarray | None = None      # tracked person position when follow == "person"
    pick: tuple[float, float] | None = None   # pending double-click (u, v in 0..1)
    send: object = None                   # callable(bytes header_json, bytes jpeg), set by the web layer


class Engine:
    def __init__(self, seq: MultiViewSequence, cams: list[str], methods: dict, fps: float = 30.0):
        self.seq, self.cams, self.methods, self.fps = seq, cams, methods, fps
        self.frames = seq.frame_ids
        self.mode = "live" if "live" in methods else next(iter(methods))
        self.playing, self.k, self.faults_on = True, 0, False
        self.viewers: dict[int, Viewer] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

        print(f"[engine] loading {len(self.frames)} frames x {len(cams)} cameras into GPU memory ...", flush=True)
        self.streams = ReplayStreams(seq, cams, self.frames, FaultConfig.parse(""), fps)
        self.sync = Synchronizer(cams, {c: (seq.camera(c).width, seq.camera(c).height) for c in cams}, fps)

        # orbit frame + limits
        self.target, self.down = scene_frame(seq)
        C = np.stack([seq.camera(c).center for c in seq.camera_names])
        rel = C - self.target
        h = rel @ self.down
        horiz = rel - np.outer(h, self.down)
        self.e1 = horiz[0] / np.linalg.norm(horiz[0])
        self.e2 = np.cross(self.down, self.e1)
        P = np.stack([horiz @ self.e1, horiz @ self.e2], 1)
        # rig footprint circle, in world coordinates (so the orbit target can move)
        self.fp_world = self.target + P.mean(0)[0] * self.e1 + P.mean(0)[1] * self.e2
        self.fp_r = np.linalg.norm(P - P.mean(0), axis=1).mean()
        self.template = seq.camera(cams[0])
        self.home = self.target.copy()
        self.state, self.fs, self.last_process_ms = None, None, 0.0
        self.pending: dict = {}

    # ---- viewer camera with limits -------------------------------------------------------
    def max_dist(self, target: np.ndarray, az_rad: float) -> float:
        """Distance from target to 90% of the rig footprint along this azimuth (stay inside)."""
        o = self.fp_world - target
        o = np.array([o @ self.e1, o @ self.e2])
        d = np.array([np.cos(az_rad), np.sin(az_rad)])
        b = d @ o
        return max(80.0, 0.9 * (b + np.sqrt(max(b * b - o @ o + self.fp_r ** 2, 0.0))))

    def people(self, frame: int) -> list[np.ndarray]:
        """Current 3D person centroids (world). Live: connected clusters of hull voxels on a 10 cm
        grid (no annotations); offline: skeletons."""
        if self.mode == "live" and isinstance(self.state, dict) and len(self.state["voxels"]):
            bg = self.methods["live"].bg
            x = self.state["voxels"].cpu().numpy() * bg.scale + bg.center
            idx = np.floor((x - x.min(0)) / 10.0).astype(int)
            grid = np.zeros(idx.max(0) + 1, bool)
            grid[tuple(idx.T)] = True
            labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
            lab = labels[tuple(idx.T)]
            out = [x[lab == i].mean(0) for i in range(1, n + 1) if (lab == i).sum() > 300]
            return out
        out = []
        for b in self.seq.bodies(frame):
            j = b["joints"][b["joints"][:, 3] > 0, :3]
            if len(j):
                out.append(j.mean(0))
        return out

    def _update_target(self, v: Viewer, people: list[np.ndarray]) -> None:
        if v.target is None:
            v.target = self.home.copy()
        goal = None
        if v.follow == "person" and v.person is not None and people:
            near = min(people, key=lambda p: np.linalg.norm(p - v.person))
            if np.linalg.norm(near - v.person) < 80:           # same person (moved < 80 cm)
                v.person = near
            goal = v.person
        elif v.follow == "group" and people:
            goal = np.mean(people, axis=0)
        if goal is not None:
            v.target = v.target + 0.2 * (goal - v.target)

    def _pick(self, v: Viewer, cam: Camera, people: list[np.ndarray]) -> None:
        """Double-click: follow the person whose centroid projects closest to the click."""
        u, w = v.pick
        v.pick = None
        if not people:
            return
        uv, depth = cam.project(np.stack(people), distort=False)
        click = np.array([u * cam.width, w * cam.height])
        d = np.linalg.norm(uv - click, axis=1) + np.where(depth > 0, 0, 1e9)
        i = int(np.argmin(d))
        if d[i] < 0.25 * cam.width:
            v.follow, v.person = "person", people[i]

    def src_density(self, target: np.ndarray) -> float:
        """Best source pixel density at the target (focal / distance): caps useful zoom."""
        return max(self.seq.camera(c).K[0, 0] / np.linalg.norm(self.seq.camera(c).center - target)
                   for c in self.cams)

    def camera(self, v: Viewer) -> tuple[Camera, dict]:
        target = v.target if v.target is not None else self.home
        az, el = np.radians(v.az), np.radians(np.clip(v.el, -5, 80))
        dist = float(np.clip(v.dist, 80.0, self.max_dist(target, az)))
        horiz = np.cos(az) * self.e1 + np.sin(az) * self.e2
        eye = target + dist * (np.cos(el) * horiz - np.sin(el) * self.down)
        R, t = look_at(eye, target, self.down)
        # zoom cap: pixel density at the target at most 3x the best real camera's (past ~1.8x the
        # view is upsampled and gets softer; the UI says so). Only limits zooming in.
        f0 = self.template.K[0, 0]
        density = self.src_density(target)
        zoom = float(np.clip(v.zoom, 0.5, max(1.0, 3.0 * density * dist / f0)))
        beyond = f0 * zoom / dist > 1.8 * density
        K = self.template.K.copy()
        K[0, 0] *= zoom
        K[1, 1] *= zoom
        cam = Camera("viewer", K, np.zeros(5), R, t, self.template.width, self.template.height)
        return cam, {"dist": round(dist), "zoom": round(zoom, 2), "zoom_limited": zoom < v.zoom - 1e-3,
                     "dist_limited": abs(dist - v.dist) > 1, "follow": v.follow, "beyond_source": bool(beyond)}

    # ---- controls (called from the web thread; applied by the engine between ticks) ------
    def set_faults(self, on: bool) -> None:
        with self.lock:
            self.pending["faults"] = bool(on)

    def seek(self, k: int) -> None:
        with self.lock:
            self.pending["seek"] = int(k) % len(self.frames)

    def set_mode(self, mode: str) -> None:
        if mode in self.methods:
            with self.lock:
                self.pending["mode"] = mode

    def _apply_pending(self) -> bool:
        """Apply queued controls; True if the current frame must be (re)processed."""
        with self.lock:
            p, self.pending = self.pending, {}
        if "faults" in p:
            self.faults_on = p["faults"]
            self.streams.faults = FaultConfig.parse(DEMO_FAULTS if p["faults"] else "")
        if "mode" in p:
            self.mode = p["mode"]
        if "seek" in p:
            self.k = p["seek"]
        return bool(p)

    # ---- main loop --------------------------------------------------------------------
    def run(self) -> None:
        tick = 1.0 / self.fps
        next_t = time.perf_counter()
        while not self.stop.is_set():
            with self.lock:
                idle = not self.viewers
            if idle:                               # nobody watching: keep the GPU free
                time.sleep(0.2)
                next_t = time.perf_counter()
                continue
            try:
                self._tick()
            except Exception:                      # never let the engine thread die silently
                traceback.print_exc()
                self.state = None
                time.sleep(0.5)
            next_t += tick
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()        # fell behind: don't try to catch up (live)

    def _tick(self) -> None:
        timer = StageTimer()
        t0 = time.perf_counter()
        changed = self._apply_pending()
        if self.playing or changed or self.state is None or self.fs is None:
            if self.playing and not changed and self.state is not None:
                self.k = (self.k + 1) % len(self.frames)
            with timer("sync_health"):
                self.fs = self.sync.assemble(self.k, self.frames[self.k], self.streams.packets(self.k))
            if self.mode == "live":
                self.state = self.methods["live"].process(self.fs.images, timer)
            else:
                self.state = "offline"
            self.last_process_ms = (time.perf_counter() - t0) * 1000

        with self.lock:
            viewers = list(self.viewers.values())
        frame = self.frames[self.k]
        people = self.people(frame)
        for v in viewers:
            t1 = time.perf_counter()
            if v.pick is not None:
                self._pick(v, self.camera(v)[0], people)
            self._update_target(v, people)
            cam, info = self.camera(v)
            if self.mode == "live":
                img = self.methods["live"].render_view(self.state, cam, timer)
            else:
                img = self.methods[self.mode].render(frame, cam, timer)
            render_ms = (time.perf_counter() - t1) * 1000
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(v.quality)])
            header = {
                "type": "frame", "slot": self.k, "frame": frame, "n": len(self.frames),
                "mode": self.mode, "playing": self.playing, "faults": self.faults_on, "people": len(people),
                "process_ms": round(self.last_process_ms, 1), "render_ms": round(render_ms, 1),
                "cams_ok": len(self.fs.images), "cams_total": len(self.cams),
                "status": [self.fs.status.get(c, "missing") for c in self.cams], **info,
            }
            if v.send:
                v.send(header, jpg.tobytes())
