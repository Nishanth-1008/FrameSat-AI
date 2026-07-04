import numpy as np
import pytest

from app.sevir.decode import (
    decode_frame,
    decode_linear,
    decode_vil,
    lght_to_grid,
    normalize_for_display,
)


def test_decode_linear_vis():
    encoded = np.array([[0, 10000], [5000, 1]], dtype=np.int16)
    decoded = decode_linear(encoded, "vis")
    np.testing.assert_allclose(decoded, encoded.astype(np.float32) * 1e-4)


def test_decode_linear_ir069_and_ir107_use_same_factor():
    encoded = np.array([[100, -200]], dtype=np.int16)
    d69 = decode_linear(encoded, "ir069")
    d107 = decode_linear(encoded, "ir107")
    np.testing.assert_allclose(d69, d107)
    np.testing.assert_allclose(d69, encoded.astype(np.float32) * 1e-2)


def test_decode_linear_rejects_vil():
    with pytest.raises(ValueError):
        decode_linear(np.zeros((2, 2), dtype=np.uint8), "vil")


def test_decode_vil_piecewise_ranges():
    x = np.array([0, 5, 6, 18, 19, 254, 255], dtype=np.uint8)
    decoded = decode_vil(x)

    # x <= 5 -> 0
    assert decoded[0] == 0.0
    assert decoded[1] == 0.0

    # 5 < x <= 18 -> (x-2)/90.66
    assert decoded[2] == pytest.approx((6 - 2) / 90.66)
    assert decoded[3] == pytest.approx((18 - 2) / 90.66)

    # 18 < x <= 254 -> exp((x-83.9)/38.9)
    assert decoded[4] == pytest.approx(np.exp((19 - 83.9) / 38.9))
    assert decoded[5] == pytest.approx(np.exp((254 - 83.9) / 38.9))

    # 255 -> missing -> NaN
    assert np.isnan(decoded[6])


def test_decode_vil_missing_sentinel_is_nan_not_zero():
    x = np.array([255, 255, 0], dtype=np.uint8)
    decoded = decode_vil(x)
    assert np.isnan(decoded[0])
    assert np.isnan(decoded[1])
    assert decoded[2] == 0.0


def test_normalize_for_display_handles_nan_and_range():
    arr = np.array([[0.0, np.nan], [10.0, 5.0]], dtype=np.float32)
    norm = normalize_for_display(arr)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    # NaN -> 0 before normalization, so it should map to the low end
    assert norm[0, 1] == pytest.approx(0.0)


def test_normalize_for_display_constant_array_returns_zeros():
    arr = np.full((4, 4), 7.0, dtype=np.float32)
    norm = normalize_for_display(arr)
    np.testing.assert_array_equal(norm, np.zeros((4, 4), dtype=np.float32))


def test_decode_frame_dispatches_by_img_type():
    vil_raw = np.array([[0, 254]], dtype=np.uint8)
    vis_raw = np.array([[0, 100]], dtype=np.int16)

    np.testing.assert_allclose(decode_frame(vil_raw, "vil"), decode_vil(vil_raw))
    np.testing.assert_allclose(decode_frame(vis_raw, "vis"), decode_linear(vis_raw, "vis"))


def test_decode_frame_rejects_lght():
    with pytest.raises(ValueError):
        decode_frame(np.zeros((2, 2)), "lght")


def test_lght_to_grid_empty_returns_zeros():
    grid = lght_to_grid(np.zeros((0, 5)), grid_size=48, n_frames=49)
    assert grid.shape == (48, 48, 49)
    assert grid.sum() == 0


def test_lght_to_grid_places_flash_in_correct_bin():
    # One flash at t=0s (frame ~24), x=10, y=20
    data = np.array([[0.0, 38.0, -97.0, 10, 20]])
    grid = lght_to_grid(data, grid_size=48, n_frames=49)
    assert grid.sum() == 1
    y, x, t = np.argwhere(grid == 1)[0]
    assert (x, y) == (10, 20)


def test_lght_to_grid_filters_out_of_bounds_points():
    data = np.array([[0.0, 38.0, -97.0, -5, 20], [0.0, 38.0, -97.0, 999, 20]])
    grid = lght_to_grid(data, grid_size=48, n_frames=49)
    assert grid.sum() == 0
