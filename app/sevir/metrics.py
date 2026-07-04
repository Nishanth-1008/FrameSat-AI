"""
Interpolation Quality Metrics

Computes similarity between a model-generated intermediate frame and the
true middle frame (available only in SEVIR mode, where the ground-truth
frame is a real observed frame from the sequence -- never available for
plain file uploads).

- PSNR / SSIM: via scikit-image (hard dependency, always available).
- LPIPS: via the `lpips` package (optional). LPIPS requires downloading a
  small pretrained network the first time it's used; if the package isn't
  installed or the weights can't be fetched (e.g. no network), we degrade
  gracefully and simply omit `lpips` from the result rather than failing
  the whole request.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS_MODEL = None
_LPIPS_UNAVAILABLE_REASON: str | None = None


@dataclass(frozen=True)
class QualityMetrics:
    psnr: float | None
    ssim: float | None
    lpips: float | None
    lpips_unavailable_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
        }


def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch computing metrics: {a.shape} != {b.shape}")


def compute_psnr(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """PSNR in dB between two uint8 HxWx3 images. Higher is better."""
    _ensure_same_shape(prediction, ground_truth)
    return float(peak_signal_noise_ratio(ground_truth, prediction, data_range=255))


def compute_ssim(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Structural similarity (0-1, higher is better) between two images."""
    _ensure_same_shape(prediction, ground_truth)
    return float(
        structural_similarity(
            ground_truth, prediction, channel_axis=-1, data_range=255
        )
    )


def _try_load_lpips():
    """Lazily load the LPIPS network. Cached; records failure reason once."""
    global _LPIPS_MODEL, _LPIPS_UNAVAILABLE_REASON

    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    if _LPIPS_UNAVAILABLE_REASON is not None:
        return None

    try:
        import lpips  # type: ignore
        import torch

        _LPIPS_MODEL = lpips.LPIPS(net="alex")
        _LPIPS_MODEL.eval()
        return _LPIPS_MODEL
    except ImportError:
        _LPIPS_UNAVAILABLE_REASON = (
            "lpips package not installed. Install with `pip install lpips` to "
            "enable perceptual similarity scoring."
        )
    except Exception as exc:  # pragma: no cover - e.g. no network for weights
        _LPIPS_UNAVAILABLE_REASON = f"LPIPS model could not be loaded: {exc}"

    return None


def compute_lpips(prediction: np.ndarray, ground_truth: np.ndarray) -> float | None:
    """
    Perceptual similarity (lower is better). Returns None if the `lpips`
    package/weights are unavailable in this environment.
    """
    _ensure_same_shape(prediction, ground_truth)
    model = _try_load_lpips()
    if model is None:
        return None

    import torch

    def to_tensor(img: np.ndarray) -> "torch.Tensor":
        # lpips expects float tensors in [-1, 1], shape (1, 3, H, W)
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t

    with torch.no_grad():
        dist = model(to_tensor(prediction), to_tensor(ground_truth))
    return float(dist.item())


def compute_all_metrics(prediction: np.ndarray, ground_truth: np.ndarray | None) -> QualityMetrics:
    """
    Compute all available quality metrics, or return an all-None result if
    no ground truth is available (e.g. plain file-upload mode).
    """
    if ground_truth is None:
        return QualityMetrics(psnr=None, ssim=None, lpips=None)

    psnr = compute_psnr(prediction, ground_truth)
    ssim = compute_ssim(prediction, ground_truth)
    lpips_score = compute_lpips(prediction, ground_truth)

    return QualityMetrics(
        psnr=psnr,
        ssim=ssim,
        lpips=lpips_score,
        lpips_unavailable_reason=_LPIPS_UNAVAILABLE_REASON,
    )
