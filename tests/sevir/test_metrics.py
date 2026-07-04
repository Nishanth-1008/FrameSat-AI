import numpy as np
import pytest

from app.sevir.metrics import compute_all_metrics, compute_psnr, compute_ssim


def _random_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)


def test_psnr_identical_images_is_infinite():
    img = _random_image()
    assert compute_psnr(img, img) == float("inf")


def test_psnr_different_images_is_finite_and_positive():
    a, b = _random_image(0), _random_image(1)
    val = compute_psnr(a, b)
    assert val > 0
    assert val != float("inf")


def test_ssim_identical_images_is_one():
    img = _random_image()
    assert compute_ssim(img, img) == pytest.approx(1.0)


def test_ssim_different_images_less_than_one():
    a, b = _random_image(0), _random_image(1)
    assert compute_ssim(a, b) < 1.0


def test_psnr_shape_mismatch_raises():
    a = _random_image()
    b = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_psnr(a, b)


def test_compute_all_metrics_without_ground_truth_returns_all_none():
    pred = _random_image()
    result = compute_all_metrics(pred, None)
    assert result.psnr is None
    assert result.ssim is None
    assert result.lpips is None


def test_compute_all_metrics_with_ground_truth_computes_psnr_ssim():
    pred, gt = _random_image(0), _random_image(1)
    result = compute_all_metrics(pred, gt)
    assert result.psnr is not None
    assert result.ssim is not None
    # lpips may or may not be available in this environment; either a float
    # or None (with a reason) is acceptable, but it must not raise.
    assert result.lpips is None or isinstance(result.lpips, float)


def test_quality_metrics_to_dict_shape():
    pred, gt = _random_image(0), _random_image(0)
    result = compute_all_metrics(pred, gt)
    d = result.to_dict()
    assert set(d.keys()) == {"psnr", "ssim", "lpips"}
