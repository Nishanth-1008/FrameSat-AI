"""
SEVIR Value Decoding

Raw SEVIR HDF5 arrays store integer-encoded values for compact storage.
This module converts them to physically meaningful floating-point values,
following the formulas in the SEVIR tutorial:

- vis / ir069 / ir107: simple linear scaling (decoded = encoded * factor)
- vil: piecewise encoding into kg/m^2, with 255 = missing
- lght: sparse (N x 5) flash matrix -> rasterized flash-count grid
"""

from __future__ import annotations

import numpy as np

from app.config import SEVIR_LINEAR_SCALE_FACTORS, SEVIR_VIL_MISSING


def decode_linear(encoded: np.ndarray, img_type: str) -> np.ndarray:
    """Decode vis/ir069/ir107 integer arrays using their linear scale factor."""
    if img_type not in SEVIR_LINEAR_SCALE_FACTORS:
        raise ValueError(
            f"decode_linear only supports {list(SEVIR_LINEAR_SCALE_FACTORS)}, got {img_type!r}"
        )
    factor = SEVIR_LINEAR_SCALE_FACTORS[img_type]
    return encoded.astype(np.float32) * factor


def decode_vil(encoded: np.ndarray) -> np.ndarray:
    """
    Decode VIL (vertically integrated liquid) integer array (0-254, 255=missing)
    into kg/m^2 using the piecewise rule from the SEVIR tutorial:

        0                          if X <= 5
        (X - 2) / 90.66            if 5 < X <= 18
        exp((X - 83.9) / 38.9)     if 18 < X <= 254
        NaN                        if X == 255 (missing)
    """
    x = encoded.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    mid = (x > 5) & (x <= 18)
    high = (x > 18) & (x <= 254)
    missing = x >= SEVIR_VIL_MISSING

    out[mid] = (x[mid] - 2) / 90.66
    out[high] = np.exp((x[high] - 83.9) / 38.9)
    out[missing] = np.nan
    # x <= 5 stays 0 (already initialized)

    return out


def normalize_for_display(decoded: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a decoded (possibly NaN-containing) array to [0, 1]
    float32, treating NaN as 0 after normalization. Useful for building the
    RIFE model's expected [0,1] float input from any SEVIR channel.
    """
    arr = np.nan_to_num(decoded.astype(np.float32), nan=0.0)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def lght_to_grid(data: np.ndarray, grid_size: int, n_frames: int) -> np.ndarray:
    """
    Convert SEVIR lightning data (Nx5 matrix: [t, lat, lon, x, y]) into an
    (grid_size, grid_size, n_frames) tensor of flash counts per pixel per
    frame, following Example 3 of the SEVIR tutorial.
    """
    frame_times = np.arange(-120.0, 125.0, 5) * 60  # seconds relative to time_utc
    out_size = (grid_size, grid_size, n_frames)

    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    x, y = data[:, 3], data[:, 4]
    mask = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[mask, :]
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    t = data[:, 0]
    z = np.digitize(t, frame_times) - 1
    z[z == -1] = 0

    xi = data[:, 3].astype(np.int64)
    yi = data[:, 4].astype(np.int64)
    zi = np.clip(z, 0, n_frames - 1).astype(np.int64)

    k = np.ravel_multi_index(np.array([yi, xi, zi]), out_size)
    counts = np.bincount(k, minlength=int(np.prod(out_size)))
    return np.reshape(counts, out_size).astype(np.float32)


def decode_frame(raw_frame: np.ndarray, img_type: str) -> np.ndarray:
    """Decode a single 2D raw frame for the given raster img_type."""
    if img_type == "vil":
        return decode_vil(raw_frame)
    if img_type in SEVIR_LINEAR_SCALE_FACTORS:
        return decode_linear(raw_frame, img_type)
    raise ValueError(f"decode_frame does not support img_type={img_type!r} (raster types only)")
