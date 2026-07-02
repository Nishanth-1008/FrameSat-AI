"""
Convert NumPy images to PyTorch tensors.
"""

import numpy as np
# pyrefly: ignore [missing-import]
import torch


class TensorConverter:
    @staticmethod
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        """
        Convert HWC NumPy image (0–1 float32)
        to BCHW PyTorch tensor.
        """

        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)   # HWC -> CHW
            .unsqueeze(0)       # CHW -> BCHW
            .float()
        )

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        return tensor