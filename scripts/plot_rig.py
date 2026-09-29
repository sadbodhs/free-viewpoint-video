"""Plot camera rig (top-down + side view) with held-out cameras and 3D people highlighted."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fvv.data import MultiViewSequence
from fvv.eval import load_or_create_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="outputs/rig")
    args = ap.parse_args()

    seq = MultiViewSequence(args.root)
    test = set(load_or_create_split(seq))
    names = seq.camera_names
    C = np.stack([seq.cameras[n].center for n in names])
    F = np.stack([seq.cameras[n].forward for n in names])
    bodies = seq.bodies(seq.frame_ids[0])
    J = np.concatenate([b["joints"][b["joints"][:, 3] > 0, :3] for b in bodies]) if bodies else np.zeros((0, 3))

    # Panoptic: y points down, floor is the x-z plane.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (a, b), title in [(axes[0], (0, 2), "top-down (x-z)"), (axes[1], (0, 1), "side (x-y)")]:
        is_test = np.array([n in test for n in names])
        ax.quiver(C[:, a], C[:, b], F[:, a], F[:, b], color=np.where(is_test, "tab:red", "tab:blue"),
                  angles="xy", scale=12, width=0.004)
        ax.scatter(C[~is_test, a], C[~is_test, b], c="tab:blue", s=25, label="train cams")
        ax.scatter(C[is_test, a], C[is_test, b], c="tab:red", s=60, label="held-out cams")
        for n, c in zip(names, C):
            ax.annotate(n, (c[a], c[b]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if len(J):
            ax.scatter(J[:, a], J[:, b], c="k", s=4, label="people (3D joints)")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("xyz"[a] + " (cm)")
        ax.set_ylabel("xyz"[b] + " (cm)")
        if b == 1:
            ax.invert_yaxis()
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{seq.root.name}: {len(names)} HD cameras")
    fig.tight_layout()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{seq.root.name}.png"
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == "__main__":
    main()
