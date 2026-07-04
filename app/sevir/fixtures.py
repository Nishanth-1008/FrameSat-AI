"""
Synthetic SEVIR Fixture Generator

Builds a small, self-contained SEVIR-shaped dataset (CATALOG.csv + HDF5
files) on disk, matching the real schema/layout described in the SEVIR
tutorial:

    <root>/CATALOG.csv
    <root>/data/<img_type>/<year>/SEVIR_<TYPE>_RANDOMEVENTS_<year>_<range>.h5

This lets us develop and unit-test the catalog/provider layer without
network access to AWS S3. It is NOT used in production -- SEVIR_ROOT should
point at a real local mirror (or S3) there. Only used by tests / local dev
seeding.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from app.config import SEVIR_FRAMES_PER_EVENT, SEVIR_RASTER_TYPES

# Small patch size for fast fixture generation/tests. Real SEVIR uses
# 384x384 (vis/ir/vil) -- we keep the same *ratio of conventions* (square,
# consistent across types for a given event) but shrink it drastically.
FIXTURE_PATCH_SIZE = 32
FIXTURE_PROJ = (
    "+proj=laea +lat_0=38 +lon_0=-98 +units=m +a=6370997.0 +b=6370997.0"
)


def _make_frames(
    img_type: str, n_events: int, size: int, n_frames: int, seed: int
) -> np.ndarray:
    """Generate plausible integer-encoded synthetic frames for an img_type."""
    rng = np.random.default_rng(seed)

    if img_type == "vil":
        # 0-254 range, with a moving "storm" blob so consecutive frames are
        # visually related (useful for sanity-checking interpolation later).
        data = np.zeros((n_events, size, size, n_frames), dtype=np.uint8)
        for e in range(n_events):
            cx, cy = rng.uniform(size * 0.3, size * 0.7, size=2)
            vx, vy = rng.uniform(-0.3, 0.3, size=2)
            for t in range(n_frames):
                yy, xx = np.mgrid[0:size, 0:size]
                blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size / 6) ** 2)))
                frame = (blob * 200).astype(np.uint8)
                data[e, :, :, t] = frame
                cx += vx
                cy += vy
        return data

    # vis / ir069 / ir107: smaller synthetic ints; exact scale doesn't matter
    # for fixtures, just that decode_linear() runs cleanly on them.
    return rng.integers(0, 4000, size=(n_events, size, size, n_frames), dtype=np.int16)


def generate_sevir_fixture(
    root: str | Path,
    n_events: int = 6,
    year: int = 2019,
    img_types: tuple[str, ...] = SEVIR_RASTER_TYPES,
    patch_size: int = FIXTURE_PATCH_SIZE,
    n_frames: int = SEVIR_FRAMES_PER_EVENT,
    overwrite: bool = True,
) -> Path:
    """
    Write a synthetic SEVIR fixture tree under `root`. Returns the path to
    the generated CATALOG.csv.
    """
    root = Path(root)
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"

    event_ids = [f"S{year}{i:04d}" for i in range(n_events)]
    base_time = pd.Timestamp(f"{year}-06-01T12:00:00Z")
    minute_offsets = ":".join(str(m) for m in range(-120, 125, 5))[: -0]  # noqa
    minute_offsets = ":".join(str(-120 + 5 * i) for i in range(n_frames))

    catalog_rows = []

    for img_type in img_types:
        file_name_rel = f"{img_type}/{year}/SEVIR_{img_type.upper()}_RANDOMEVENTS_{year}_0101_1231.h5"
        file_path = data_dir / file_name_rel
        file_path.parent.mkdir(parents=True, exist_ok=True)

        frames = _make_frames(
            img_type, n_events, patch_size, n_frames, seed=hash(img_type) % (2**31)
        )

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("id", data=np.array(event_ids, dtype="S32"))
            hf.create_dataset(img_type, data=frames)

        for idx, event_id in enumerate(event_ids):
            offset_time = base_time + pd.Timedelta(minutes=idx)  # stagger slightly
            catalog_rows.append(
                {
                    "id": event_id,
                    "file_name": file_name_rel,
                    "file_index": idx,
                    "img_type": img_type,
                    "time_utc": offset_time,
                    "minute_offsets": minute_offsets,
                    "episode_id": 100000 + idx,
                    "event_id": 200000 + idx,
                    "llcrnrlat": 30.0 + idx * 0.5,
                    "llcrnrlon": -100.0 + idx * 0.5,
                    "urcrnrlat": 33.0 + idx * 0.5,
                    "urcrnrlon": -97.0 + idx * 0.5,
                    "proj": FIXTURE_PROJ,
                    "size_x": patch_size,
                    "size_y": patch_size,
                    "height_m": 384000,
                    "width_m": 384000,
                    "data_min": 0,
                    "data_max": 254 if img_type == "vil" else 4000,
                    "pct_missing": 0.0,
                }
            )

    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = root / "CATALOG.csv"
    catalog_df.to_csv(catalog_path, index=False)
    return catalog_path


if __name__ == "__main__":
    import sys

    out = generate_sevir_fixture(sys.argv[1] if len(sys.argv) > 1 else "sevir_data_fixture")
    print(f"Wrote synthetic SEVIR fixture catalog to {out}")
