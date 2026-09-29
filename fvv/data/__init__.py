from .camera import Camera
from .sequence import MultiViewSequence, load_cameras, save_cameras

# COCO-19 skeleton used by CMU Panoptic (0-indexed joint pairs).
COCO19_EDGES = [(0, 1), (0, 3), (3, 4), (4, 5), (0, 2), (2, 6), (6, 7), (7, 8),
                (2, 12), (12, 13), (13, 14), (0, 9), (9, 10), (10, 11)]
