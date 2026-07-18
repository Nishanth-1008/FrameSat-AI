# FrameSat-AI Kaggle Training

This bundle contains everything required to run the standalone Practical-RIFE 4.26 training framework on Kaggle.

## Setup Instructions

1. **Upload Dataset**: Upload this `kaggle_bundle.zip` as a Kaggle Dataset, or directly upload and extract it in your notebook's working directory (`/kaggle/working/`).
2. **Mount SEVIR Data**: Ensure the SEVIR input dataset is added to your Kaggle notebook (e.g. `/kaggle/input/sevir-dataset/`).
3. **Install Dependencies**: Run the setup script in a notebook cell:
   ```bash
   !bash kaggle/setup.sh
   ```

## Training

To begin training, simply run:
```bash
!python training/train.py --config kaggle/train_kaggle.json
```

**Outputs**:
All outputs, including checkpoints (`best.pth`, `latest.pth`), metrics (`metrics.csv`), validation visual examples, and TensorBoard logs, will be automatically written to the working directory at `/kaggle/working/artifacts/training/runs/run_<timestamp>_<experiment_name>/`. You can simply download this directory or the entire `/kaggle/working/` folder after training completes.

## Resuming from Checkpoints
To resume training from a previously downloaded checkpoint:
1. Upload your checkpoint `.pth` file as a Kaggle dataset.
2. Edit `kaggle/train_kaggle.json` and set `"resume_checkpoint": "/kaggle/input/your-checkpoint-dataset/best.pth"`.
3. Set `"resume": true`.
