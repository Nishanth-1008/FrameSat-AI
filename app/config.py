"""
Application configuration.
"""

from pathlib import Path
# pyrefly: ignore [missing-import]
import torch

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# External RIFE Repository
RIFE_ROOT = PROJECT_ROOT.parent / "Practical-RIFE"

# Model Weights
MODEL_DIR = RIFE_ROOT / "train_log"

# Default inference size (Width, Height)
IMAGE_SIZE = (448, 256)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# SEVIR Dataset Configuration
# ---------------------------------------------------------------------------
#
# SEVIR ("Storm EVent ImagRy") is a public dataset of aligned GOES-16 / NEXRAD
# imagery released as an AWS Open Data set. It is organized as:
#
#   - CATALOG.csv  -- one row per (event, img_type) describing where the
#                      event's frames live (file_name/file_index) plus
#                      georeferencing + storm metadata.
#   - data/<img_type>/<year>/SEVIR_<TYPE>_..._<range>.h5 -- HDF5 files,
#                      each containing many events keyed by row index, with
#                      a dataset per event of shape (L, L, T).
#
# Reference: SEVIR_Tutorial.ipynb / https://github.com/MIT-AI-Accelerator/eie-sevir
#
# In this sandbox we cannot reach S3, so SEVIR_DATA_DIR defaults to a local
# directory that (in dev/test) is populated with small synthetic fixtures
# mirroring the real layout. On the user's machine this should be pointed at
# a real local mirror of `s3://sevir` (see download instructions in the
# tutorial), or SEVIR_USE_S3=1 can be set to read directly via fsspec/s3fs.

import os

SEVIR_ROOT = Path(os.environ.get("SEVIR_ROOT", str(PROJECT_ROOT / "sevir_data")))
SEVIR_CATALOG_PATH = Path(
    os.environ.get("SEVIR_CATALOG_PATH", str(SEVIR_ROOT / "CATALOG.csv"))
)
SEVIR_DATA_DIR = Path(os.environ.get("SEVIR_DATA_DIR", str(SEVIR_ROOT / "data")))

# Whether to read HDF5 files from S3 (s3://sevir) instead of SEVIR_DATA_DIR.
# Requires network access + `s3fs`/`fsspec`, unavailable in this sandbox.
SEVIR_USE_S3 = os.environ.get("SEVIR_USE_S3", "0") == "1"
SEVIR_S3_BUCKET = os.environ.get("SEVIR_S3_BUCKET", "sevir")

# Each SEVIR event spans 4 hours at 5-minute cadence -> 49 frames for raster
# types (vis/ir069/ir107/vil). `lght` is a sparse point-event type, handled
# separately (rasterized on demand).
SEVIR_FRAMES_PER_EVENT = 49
SEVIR_FRAME_INTERVAL_MINUTES = 5

# Image (sensor) types available in SEVIR, and which are "raster" types with
# a fixed (L, L, T) tensor in the HDF5 file vs. the sparse `lght` type.
SEVIR_RASTER_TYPES = ("vis", "ir069", "ir107", "vil")
SEVIR_ALL_TYPES = SEVIR_RASTER_TYPES + ("lght",)

# Default image type used to build the preview/interpolation input when the
# user doesn't specify one. `vil` (vertically integrated liquid) is the most
# visually informative single-channel product for storm structure.
SEVIR_DEFAULT_IMG_TYPE = "vil"

# Linear decoding scale factors for satellite channels (see tutorial Table 3).
# decoded = encoded * SCALING_FACTOR
SEVIR_LINEAR_SCALE_FACTORS = {
    "vis": 1e-4,    # -> reflectance factor
    "ir069": 1e-2,  # -> degrees C
    "ir107": 1e-2,  # -> degrees C
}

# `vil` uses a piecewise encoding (not a simple linear scale) into kg/m^2,
# with 255 reserved as a missing-data sentinel. See SEVIR_VIL_MISSING and
# app/sevir/decode.py for the decode formula.
SEVIR_VIL_MISSING = 255

# Max number of events returned by /datasets/sevir/events without paging.
SEVIR_EVENTS_DEFAULT_PAGE_SIZE = 50
SEVIR_EVENTS_MAX_PAGE_SIZE = 200