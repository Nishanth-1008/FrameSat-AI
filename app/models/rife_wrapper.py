"""
Wrapper around Practical-RIFE.

The rest of GeoVFI should only interact with this class.
"""

import sys
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.nn import functional as F
from pathlib import Path

from app.config import RIFE_ROOT, MODEL_DIR

# Allow Python to import Practical-RIFE modules
sys.path.insert(0, str(RIFE_ROOT))

try:
    from train_log.RIFE_HDv3 import Model  # type: ignore
    HAS_RIFE = True
except ImportError:
    HAS_RIFE = False


class RIFEWrapper:
    """Loads and manages the Practical-RIFE model, with fallback stub."""

    def __init__(self):
        if HAS_RIFE:
            self.model = Model()
            self.model.load_model(str(MODEL_DIR), -1)
            self.model.eval()
            self.model.device()
        else:
            print("WARNING: Practical-RIFE is not installed/available. Using stub interpolation.")
            self.model = None

    def interpolate(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if self.model is None:
            # Fallback stub: return the first frame directly
            return img1

        _, _, h, w = img1.shape

        # Pad to multiples of 128 (required by Practical-RIFE)
        ph = ((h - 1) // 128 + 1) * 128
        pw = ((w - 1) // 128 + 1) * 128

        padding = (0, pw - w, 0, ph - h)

        img1 = F.pad(img1, padding)
        img2 = F.pad(img2, padding)

        with torch.no_grad():
            output = self.model.inference(
                img1,
                img2,
                timestep=timestep,
                scale=scale,
            )

        # Crop back to original size
        output = output[:, :, :h, :w]

        return output


    def get_model(self):
        """Return the loaded model."""
        return self.model