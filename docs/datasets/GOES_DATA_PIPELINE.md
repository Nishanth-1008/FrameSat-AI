# GOES-19 Dataset Acquisition and QA Pipeline

This document details the architecture, configuration, operation, and scaling guidelines for the GOES-19 Dataset Acquisition Pipeline. The system is designed to acquire, cache, index, validate, and verify ABI Channel 13 satellite scenes for training Frame Interpolation models.

## 1. System Architecture

The pipeline consists of modular subsystems:
- **`downloader.py`**: Queries NOAA's S3 bucket asynchronously with concurrent workers and automatic exponential retries.
- **`validator.py`**: Performs MD5 hashing, NetCDF structure checking, Planck coefficient extraction, and timestamps verification. Isolates failed files in a quarantine folder.
- **`db.py`**: Indexes valid scenes in a SQLite database `metadata.db` to prevent repetitive file scans.
- **`stats_generator.py`**: Calculates scene counts, temporal coverages, file sizes, cadences, and temperature histograms.
- **`qa_runner.py`**: Coordinates post-download scientific validation, timeline checks, visual GIF/PNG triplet generation, and outputs an HTML Manual Review gallery.
- **`pipeline.py`**: Orchestrates downloading, validation, statistics, and QA loops. Integrates a **Target Size Controller** to download until the target count is satisfied.

## 2. Directory Structure

```
FrameSat-AI/
├── download_goes.py             # Root execution CLI wrapper
├── export_goes_dataset.py       # Zips valid scenes & metadata for Kaggle
├── pipeline_config.json         # Master configuration parameters
├── datasets/
│   ├── goes19_cache/            # Locally cached NetCDF .nc files
│   ├── goes19_quarantine/       # Damaged/corrupt files isolated here
│   └── metadata.db              # SQLite index of active valid scenes
└── training/data/
    ├── goes19/
    │   ├── db.py                # Database manager
    │   ├── downloader.py        # Asynchronous downloader
    │   ├── validator.py         # File checker & quarantine coordinator
    │   ├── stats_generator.py   # Statistics calculator
    │   ├── qa_runner.py         # QA pipeline wrapper
    │   └── pipeline.py          # Master orchestrator
    └── qa/
        └── reports/             # QA reports & Visual review assets
```

## 3. SQLite Metadata Schema

Database table: `scenes`
- `scene_id` (TEXT PRIMARY KEY) - Unique scene string (derived from base filename).
- `timestamp` (TEXT) - ISO-formatted start timestamp.
- `satellite` (TEXT) - Satellite identifier (e.g. `GOES19`).
- `channel` (INTEGER) - ABI band channel (default: `13`).
- `sector` (TEXT) - Observation sector (`CONUS`, `Full Disk`, `Mesoscale 1`, `Mesoscale 2`).
- `scan_mode` (TEXT) - Sensor scan speed mode (e.g. `M6`).
- `filepath` (TEXT) - Absolute path to cached NetCDF file.
- `filesize` (INTEGER) - File size in bytes.
- `checksum` (TEXT) - MD5 checksum hash.
- `download_time` (TEXT) - Download execution timestamp.

## 4. Workflows

### Download & Validation Flow
1. S3 bucket keys are fetched hourly starting chronologically from `start_date`.
2. Available keys are filtered to match the sector and channel.
3. Checks if the file already exists locally, matches S3 file size, and has a valid record in `metadata.db` to skip downloading.
4. Downloads NetCDF file asynchronously with retries on failure.
5. Post-download checks are run:
   - MD5 hash calculation.
   - Readability of NetCDF variables (`Rad`, `planck_fk1`, `planck_fk2`, etc.).
   - Correct structure and non-corrupt values.
6. Non-corrupted files are indexed in `metadata.db`. Damaged files are moved to `datasets/quarantine/goes19_quarantine/` alongside a `*.reason.txt` metadata file detailing the failure.

### QA Execution Flow
On completion, the QA orchestrator automatically executes:
1. **Scientific Verification**: Validates Planck FK1/FK2 and BC1/BC2 coefficients and confirms physically consistent brightness temperatures (180K–320K).
2. **Dataset Integrity**: Checks chronological sequence of timestamps, flags gaps, and outputs time interval histograms.
3. **Visual QA**: Extracts triplets `(t0, t1, t2)` from the database, renders side-by-side PNGs with delta annotations, compiles animated GIFs cycling through the frames, and links them inside a single `visual_qa_gallery.html` webpage.

## 5. Usage & Configuration

Modify parameters inside `pipeline_config.json` or override them directly via command line arguments.

### Quick Start: Run Acquisition
```bash
python download_goes.py --target-scenes 1000 --workers 8
```

### Command Line Arguments
- `--config`: Custom JSON configuration path.
- `--target-scenes`: Target count of valid scenes to acquire.
- `--sector`: S3 GOES-19 Sector (`CONUS`, `Full Disk`, `Mesoscale 1`, `Mesoscale 2`).
- `--channel`: Band channel (default: `13`).
- `--workers`: Concurrent download thread pool workers count.
- `--start-date`: Acquisition starting timestamp (ISO format, e.g. `2024-10-10T21:00:00`).
- `--no-verify`: Skips validation check post-download.
- `--no-qa`: Skips running automated QA post-download.

### Packing for Kaggle
Generate the uploadable dataset zip package by running:
```bash
python export_goes_dataset.py
```
This produces `framesat-goes19-v1.zip` matching Kaggle requirements.

## 6. Guidelines for Scaling (5k, 10k, 20k Scenes)

When expanding the dataset, consider the following performance guidelines:

1. **Storage Footprint**:
   - `CONUS` (1500×2500 pixels) NetCDF files are roughly ~2.4MB per scene.
     - **5k scenes**: ~12 GB
     - **10k scenes**: ~24 GB
     - **20k scenes**: ~48 GB
   - Ensure the training system disk has enough free capacity.
2. **Workers Constraint**:
   - For 10k+ scenes, increase thread workers (e.g. `--workers 16` or `32` if you have high network bandwidth) to maximize download speeds.
3. **Database Maintenance**:
   - The database indexes scenes in millisecond speeds, avoiding slow directory lookups when processing large scales.
4. **Acquisition Range Calculation**:
   - Simply change `--target-scenes` to `5000` or `10000`. The downloader will automatically increment chronological dates, query S3, validate, and append new scenes to the cache without double-downloading or rescanning directories.
