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