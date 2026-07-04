# SEVIR Dataset Integration

This document explains how the SEVIR dataset browsing feature is wired up,
and how to point it at **real** SEVIR data on a machine that has AWS S3
access (this sandbox does not, so everything here was built and tested
against synthetic fixtures).

## Architecture

```
app/sevir/
  catalog.py    -- loads & validates CATALOG.csv, groups rows into events
  decode.py     -- integer -> physical value decoding (vis/ir/vil/lght)
  provider.py   -- SEVIRProvider: catalog + HDF5 reads -> RGB-ish frames
  metrics.py    -- PSNR / SSIM / LPIPS between prediction and ground truth
  fixtures.py   -- synthetic CATALOG.csv + HDF5 generator (dev/test only)

backend_api/
  api.py           -- existing file-upload /interpolate endpoint (unchanged)
  sevir_routes.py  -- /datasets, /datasets/sevir/events[...], /interpolate/sevir

tests/sevir/   -- pytest suite (56 tests) run entirely against fixtures
```

`SEVIRProvider` is the single point of contact with SEVIR data, mirroring
the `ImageProvider` (uploads) / `SEVIRProvider` (dataset) split described in
the project's research report. It never assumes fixture data -- it reads
whatever `CATALOG.csv` / HDF5 files are actually present at
`SEVIR_CATALOG_PATH` / `SEVIR_DATA_DIR`.

## Running against real SEVIR data

1. Download the catalog and the image types you want (e.g. `vil`):

   ```bash
   aws s3 cp --no-sign-request s3://sevir/CATALOG.csv /path/to/sevir_data/CATALOG.csv
   aws s3 sync --no-sign-request s3://sevir/data/vil /path/to/sevir_data/data/vil
   ```

   (See `SEVIR_Tutorial.ipynb` for the full download instructions and
   dataset description.)

2. Point the backend at it, e.g. via `.env` or shell exports:

   ```bash
   export SEVIR_ROOT=/path/to/sevir_data
   # or individually:
   export SEVIR_CATALOG_PATH=/path/to/sevir_data/CATALOG.csv
   export SEVIR_DATA_DIR=/path/to/sevir_data/data
   ```

3. Start the backend as usual (`uvicorn backend_api.api:app`). The existing
   `/datasets/sevir/events` etc. endpoints will now serve real events.

No code changes are required -- `SEVIRProvider` and the catalog loader were
built directly against the real CATALOG.csv schema (see `app/sevir/catalog.py`
docstring) and HDF5 layout (`N x L x L x T` per img_type, `id` dataset for
event lookup), verified against `SEVIR_Tutorial.ipynb` and the
`MIT-AI-Accelerator/eie-sevir` repo conventions.

## What's synthetic vs. real-schema-accurate

- **Real-schema-accurate**: catalog columns, HDF5 file layout, vis/ir069/ir107
  linear decoding, VIL piecewise decoding, minute_offsets parsing (49 frames
  at 5-minute cadence), lightning-to-grid rasterization.
- **Synthetic (dev/test only)**: the actual pixel values and event count in
  `app/sevir/fixtures.py` -- these exist purely so the provider/API layer can
  be exercised by `pytest` without network access. `generate_sevir_fixture()`
  is never imported by production code paths (`backend_api/api.py`,
  `backend_api/sevir_routes.py`), only by `tests/sevir/conftest.py`.

## Known gap: end-to-end run with real RIFE weights

`InterpolationService` (and therefore `POST /interpolate/sevir`) requires
the external `Practical-RIFE` repo + pretrained weights to sit alongside
this project (see `app/config.py: RIFE_ROOT`). That's unavailable in this
sandbox, so:

- `tests/sevir/test_api_routes.py` exercises the SEVIR routes with a mock
  interpolation service (copies frame A to the output path) to validate the
  request/response contract, metrics wiring, and error handling.
- The real end-to-end run (actual RIFE inference on real SEVIR frames)
  should be done on your machine once Practical-RIFE is set up, per the
  existing `app/config.py: RIFE_ROOT` convention.

## Quality metrics

`POST /interpolate/sevir` computes PSNR/SSIM whenever `frame_a` and
`frame_b` are exactly 2 frames apart (so the true middle frame exists in
the sequence and is used as ground truth). LPIPS is computed too if the
`lpips` package + weights are available; otherwise it's returned as `null`
without failing the request (see `app/sevir/metrics.py`).

Frames that aren't 2 apart still interpolate normally -- they just have no
ground truth, so `psnr`/`ssim`/`lpips`/`ground_truth_frame` are `null`,
matching plain file-upload behavior.
