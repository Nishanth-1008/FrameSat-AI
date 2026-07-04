"""
Data Loader Module

Responsible for:
- Loading satellite images from disk
- Validating file existence
- Returning NumPy arrays
"""

from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2 
import numpy as np


class DataLoader:
    """Utility class for loading image files."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image.

        Returns:
            NumPy array containing the image.

        Raises:
            FileNotFoundError
            ValueError
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    @staticmethod
    def load_pair(frame1: str, frame2: str):
        """
        Load two images.

        Returns:
            Tuple[np.ndarray, np.ndarray]
        """

        return (
            DataLoader.load_image(frame1),
            DataLoader.load_image(frame2),
        )