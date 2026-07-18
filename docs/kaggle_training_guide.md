# Kaggle Training Launch Guide

This document covers everything needed to launch a controlled training run of FrameSat-AI on Kaggle from a clean checkout.

## Prerequisites

You need two datasets added to your Kaggle notebook before running:

| Dataset | Purpose |
|---|---|
| `framesat-ai-training-bundle` | The code bundle (this repository, zipped) |
| `framesat-goes19-v1` | The GOES-19 NetCDF scene cache |

## Directory Convention

| Location | Contents | Writable |
|---|---|---|
| `/kaggle/input/...` | Source datasets (read-only) | ✗ |
| `/kaggle/working/` | All outputs (checkpoints, logs, metadata) | ✓ |

> [!IMPORTANT]
> **Never write into `/kaggle/input`.** The code enforces this automatically through
> the `METADATA_DIR` env var and priority-ordered path resolution in `GOES19TripletDataset`.

## Environment Variables

Set these at the top of the Configuration Cell (Section 5) in the notebook:

```python
import os

os.environ["METADATA_DIR"] = "/kaggle/working/goes19_metadata"
os.environ["CACHE_DIR"]    = "/kaggle/working/datasets/smoke_cache"   # smoke only
os.environ["CKPT_DIR"]     = "/kaggle/working/checkpoints"
os.environ["LOG_DIR"]      = "/kaggle/working/logs"
```

These are all optional — if not set, the code auto-detects Kaggle and uses the correct defaults. Setting them explicitly makes the paths visible and reproducible.

## First Training Run

The notebook uses `kaggle/train_kaggle_first.json` for the first run. Key parameters:

```json
{
  "epochs": 2,
  "batch_size": 1,
  "num_workers": 2,
  "max_triplets": 200
}
```

`max_triplets: 200` means only 200 randomly selected triplets are used for training (and ~40 for validation). This allows the first run to complete in under 30 minutes on a T4 GPU, validating the entire pipeline before committing to a full run.

To scale up for subsequent runs, increase these values in `kaggle/train_kaggle.json`:

```json
{
  "epochs": 10,
  "batch_size": 4,
  "num_workers": 4,
  "max_triplets": null
}
```

## Expected Output Structure

After a successful run, `/kaggle/working/` will contain:

```
/kaggle/working/
├── goes19_metadata/
│   ├── triplet_metadata_train.json    # train split index
│   └── triplet_metadata_val.json      # val split index
├── artifacts/training/runs/
│   └── Experiment_001/
│       ├── config.json                # resolved config snapshot
│       ├── training.log               # stdout + stderr
│       ├── metrics.csv                # epoch-by-epoch metrics
│       ├── metrics.json
│       ├── tensorboard/               # TensorBoard event files
│       ├── sample_predictions/        # 10 random val triplet visualizations
│       ├── latest.pth                 # most recent checkpoint
│       ├── best.pth                   # best checkpoint by PSNR
│       └── environment.json           # GPU/CUDA/PyTorch metadata
└── experiment_001_outputs.tar.gz      # compressed export (Section 12)
```

## Resuming from a Checkpoint

To resume a previously interrupted run:

1. Add the previous run's output as a Kaggle dataset, or upload `latest.pth` manually.
2. In the config, set:

```json
{
  "resume": true,
  "resume_checkpoint": "/kaggle/working/artifacts/training/runs/Experiment_001/latest.pth"
}
```

The trainer will automatically load the checkpoint and continue from `epoch + 1`.

## Pre-Flight Checks (Section 6)

Before training begins, the notebook runs four checks:

| Check | What it validates |
|---|---|
| Pretrained weights | `flownet.pkl` is accessible inside the bundle |
| Triplet count | At least 3 valid scenes form at least 1 triplet |
| Disk space | ≥ 5 GB free in `/kaggle/working` |
| Write access | `/kaggle/working` is writable |

If any check fails, the notebook aborts before training begins.

## Troubleshooting

### `OSError: [Errno 30] Read-only file system`
This means metadata was being written into `/kaggle/input`. Fixed by the `METADATA_DIR` priority chain. Ensure you are using the latest version of the bundle and that `METADATA_DIR` is set in the configuration cell.

### `Abort training: Pretrained weights not found`
The bundle does not include `flownet.pkl`. Ensure `weights/rife_426/train_log/flownet.pkl` is present in the uploaded bundle zip. Check the pre-flight output for the resolved path.

### `Not enough files downloaded to form a triplet`
The dataset discovery failed. Check the dataset is added to the notebook and `/kaggle/input` contains a `cache` subdirectory with `.nc` files.

### TensorBoard
To view live metrics during training:
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/artifacts/training/runs
```
