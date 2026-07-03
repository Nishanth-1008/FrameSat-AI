from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.getenv(
    "MODEL_WEIGHTS_PATH",
    str(BASE_DIR / "weights" / "rife.pth")
)

INPUT_DIR = BASE_DIR / "assets" / "sample_inputs"

OUTPUT_DIR = BASE_DIR / "assets" / "sample_outputs"

DEVICE = os.getenv("DEVICE", "cpu")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

IMAGE_SIZE = (512, 512)

MAX_UPLOAD_SIZE_MB = 25

# Integration configurations (retained for backward compatibility)
PROJECT_ROOT = BASE_DIR
RIFE_ROOT = PROJECT_ROOT.parent / "Practical-RIFE"
MODEL_DIR = RIFE_ROOT / "train_log"