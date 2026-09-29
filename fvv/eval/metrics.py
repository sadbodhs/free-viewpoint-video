"""Image quality metrics for novel-view evaluation. Images are RGB uint8 (H, W, 3)."""
import numpy as np
import torch
from skimage.metrics import structural_similarity

_lpips = None


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (pred.astype(np.float64) - gt.astype(np.float64)) / 255.0
    if mask is not None:
        diff = diff[mask.astype(bool)]
    mse = np.mean(diff ** 2)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def color_correct(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Best affine color transform (3x4, least squares) mapping pred onto gt.

    Cameras in a rig are rarely color-matched; a novel view cannot know the held-out camera's
    exposure/white balance. PSNR after this correction isolates geometry/texture errors.
    """
    x = pred.reshape(-1, 3).astype(np.float64) / 255
    y = gt.reshape(-1, 3).astype(np.float64) / 255
    X = np.concatenate([x, np.ones((len(x), 1))], 1)
    A, *_ = np.linalg.lstsq(X, y, rcond=None)
    return (np.clip(X @ A, 0, 1) * 255).round().astype(np.uint8).reshape(pred.shape)


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(structural_similarity(pred, gt, channel_axis=2, data_range=255))


@torch.no_grad()
def lpips(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> float:
    global _lpips
    if _lpips is None:
        import lpips as lp
        _lpips = lp.LPIPS(net="alex", verbose=False).to(device).eval()
    to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
    return float(_lpips(to_t(pred), to_t(gt)))


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {"psnr": psnr(pred, gt), "psnr_cc": psnr(color_correct(pred, gt), gt),
            "ssim": ssim(pred, gt), "lpips": lpips(pred, gt)}
