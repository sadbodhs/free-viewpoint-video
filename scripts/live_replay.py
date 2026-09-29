"""R0: replay a recorded capture as live camera streams through an online method.

Simulates RTSP-style delivery (latency, jitter, faults), assembles frame sets on a fixed
clock without waiting for late cameras, runs the method's per-frame processing and renders
the viewer's camera (shared orbit path). Processing time is measured on the GPU; if a frame
takes longer than the frame budget, the following slots are dropped, as in a live system.

    scripts/run.sh python scripts/live_replay.py data/panoptic/170221_haggling_b1 --method hull
    scripts/run.sh python scripts/live_replay.py data/panoptic/170221_haggling_b1 --method hull \
        --faults "jitter=40,drop=00_05@1-3,freeze=00_07@2-3,blur=00_03@0.5-4,corrupt=00_10@1.5,res=00_04@2x0.5"

Outputs outputs/live/<seq>/<method>[_<tag>].mp4 (view + camera health strip) and .json report.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fvv.data import MultiViewSequence
from fvv.eval import StageTimer, load_or_create_split
from fvv.eval.metrics import color_correct, psnr
from fvv.live import FaultConfig, ReplayStreams, Synchronizer
from fvv.methods import make_method
from fvv.render import MiniMap, VideoWriter, even, inset, label, orbit_path, scene_frame

STATUS_COLOR = {"ok": (60, 200, 90), "resized": (90, 170, 255), "late": (255, 200, 40), "missing": (230, 50, 50),
                "corrupt": (200, 60, 200), "frozen": (255, 140, 0), "blank": (120, 120, 120), "blur": (255, 110, 160)}


def health_strip(cams: list[str], status: dict[str, str], width: int) -> np.ndarray:
    strip = np.full((26, width, 3), 20, np.uint8)
    w = width / len(cams)
    for i, c in enumerate(cams):
        x0 = int(i * w) + 2
        cv2.rectangle(strip, (x0, 4), (int((i + 1) * w) - 2, 21), STATUS_COLOR[status.get(c, "missing")], -1)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--method", default="hull")
    ap.add_argument("--faults", default="", help="see fvv/live/faults.py")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--buffer-ms", type=float, default=100.0, help="jitter buffer / slot deadline")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5, help="score held-out cameras every N slots (0 = off)")
    ap.add_argument("--out", default="outputs/live")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root, undistort=True, scale=args.scale)
    test_cams = load_or_create_split(seq)
    cams = [c for c in seq.camera_names if c not in test_cams]
    frames = seq.frame_ids[: int(args.seconds * args.fps)]
    faults = FaultConfig.parse(args.faults)

    method = make_method(args.method, ckpt_dir=f"outputs/models/{seq.root.name}")
    method.fit(seq, cams, frames)
    print(f"preloading {len(frames)} frames x {len(cams)} cameras ...", flush=True)
    streams = ReplayStreams(seq, cams, frames, faults, args.fps)
    sync = Synchronizer(cams, {c: (seq.camera(c).width, seq.camera(c).height) for c in cams},
                        args.fps, args.buffer_ms)

    template = seq.camera(cams[0])
    path = orbit_path(seq, template, n=len(frames))
    _, down = scene_frame(seq)
    minimap = MiniMap([seq.camera(c) for c in seq.camera_names], down, size=180)
    budget_ms = 1000 / args.fps

    # warm-up outside the measured run (CUDA init, kernel compilation)
    method.render_view(method.process({c: streams.cache[(c, frames[0])] for c in cams}, StageTimer()),
                       path[0], StageTimer())

    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.out) / seq.root.name
    out.mkdir(parents=True, exist_ok=True)
    W, H = even(np.zeros((template.height + 26, template.width))).shape[::-1]
    timer = StageTimer()
    busy_until, dropped, last_img = 0.0, 0, None
    status_counts, evals, slot_ms = Counter(), [], []
    with VideoWriter(str(out / f"{args.method}{tag}.mp4"), W, H, fps=args.fps) as vw:
        for k, f in enumerate(frames):
            t = k / args.fps
            fs = sync.assemble(k, f, streams.packets(k))
            status_counts.update(fs.status.values())
            if t < busy_until and last_img is not None:        # pipeline still busy: slot dropped
                dropped += 1
                img = last_img.copy()
                label(img, f"DROPPED slot {k}", pos="top")
            else:
                n_before = {s: len(v) for s, v in timer.times.items()}
                state = method.process(fs.images, timer)
                img = method.render_view(state, path[k], timer)
                ms = sum(v[-1] for s, v in timer.times.items() if len(v) > n_before.get(s, 0))
                slot_ms.append(ms)
                busy_until = t + ms / 1000
                last_img = img
                if args.eval_every and k % args.eval_every == 0:
                    for c in test_cams:
                        if seq.is_valid(c, f):
                            gt = seq.image(c, f)
                            pred = method.render_view(state, seq.camera(c), StageTimer())
                            evals.append({"slot": k, "cam": c, "psnr": psnr(pred, gt),
                                          "psnr_cc": psnr(color_correct(pred, gt), gt)})
                img = img.copy()
                label(img, f"{args.method} | slot {k} | {ms:.0f} ms (budget {budget_ms:.0f}) | "
                           f"{len(fs.images)}/{len(cams)} cams", pos="bottom")
            inset(img, minimap.draw(path[k], highlight=set(test_cams)))
            frame = even(np.concatenate([img, health_strip(cams, fs.status, img.shape[1])], 0))
            vw.write(frame)

    lat = timer.summary()
    report = {
        "method": args.method, "faults": args.faults, "fps": args.fps, "buffer_ms": args.buffer_ms,
        "slots": len(frames), "dropped_slots": dropped,
        "slot_ms": {"mean": float(np.mean(slot_ms)), "p95": float(np.percentile(slot_ms, 95)),
                    "max": float(np.max(slot_ms))},
        "stages_ms": {s: {"mean": round(v["mean_ms"], 2), "p95": round(v["p95_ms"], 2)} for s, v in lat.items()},
        "camera_status": dict(status_counts),
        "heldout": {"psnr": float(np.mean([e["psnr"] for e in evals])) if evals else None,
                    "psnr_cc": float(np.mean([e["psnr_cc"] for e in evals])) if evals else None,
                    "n": len(evals)},
        "glass_to_glass_ms_estimate": args.buffer_ms + float(np.mean(slot_ms)),
    }
    (out / f"{args.method}{tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
