"""
Image Preprocessing Module

Prepares images for AI inference.
"""

# pyrefly: ignore [missing-import]
import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocess images before interpolation."""

    @staticmethod
    def preprocess(
        img: np.ndarray,
        size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Preprocess an image.

        Args:
            img: Input image.
            size: Optional (width, height).

        Returns:
            Preprocessed image.
        """

        # Resize if requested
        if size is not None:
            img = cv2.resize(img, size)

        # Convert OpenCV BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img