"""
Image Validation Module

Validates image pairs before preprocessing and inference.
"""

import numpy as np


class ImageValidator:
    """Validates loaded images."""

    @staticmethod
    def validate(img1: np.ndarray, img2: np.ndarray) -> None:
        """Raise an exception if the image pair is invalid."""

        if img1 is None or img2 is None:
            raise ValueError("One or both images are None.")

        if img1.shape != img2.shape:
            raise ValueError(
                f"Image dimensions do not match: "
                f"{img1.shape} != {img2.shape}"
            )

        if len(img1.shape) != 3:
            raise ValueError("Images must have 3 dimensions (H, W, C).")

        if img1.shape[2] != 3:
            raise ValueError("Images must have exactly 3 color channels.")