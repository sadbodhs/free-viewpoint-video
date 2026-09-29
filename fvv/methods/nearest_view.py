"""Floor baseline: show the closest training camera's image, no reconstruction.

Every real method must beat this; it is also what "just switch cameras" looks like.
"""
import numpy as np

from fvv.data import Camera, MultiViewSequence
from fvv.eval import StageTimer


class NearestView:
    name = "nearest_view"

    def fit(self, seq: MultiViewSequence, train_cams: list[str], frames: list[int]) -> None:
        self.seq = seq
        self.train = [seq.camera(c) for c in train_cams]

    def nearest(self, camera: Camera, frame: int) -> Camera:
        # Mix position and viewing direction so cameras facing away are not picked.
        cands = [c for c in self.train if self.seq.is_valid(c.name, frame)]
        score = [np.linalg.norm(c.center - camera.center) / 100.0
                 + 2.0 * (1 - c.forward @ camera.forward) for c in cands]
        return cands[int(np.argmin(score))]

    def render(self, frame: int, camera: Camera, timer: StageTimer) -> np.ndarray:
        with timer("select"):
            src = self.nearest(camera, frame)
        with timer("load"):
            return self.seq.image(src.name, frame)
