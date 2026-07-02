"""
Post-processing utilities for RIFE output.
"""

# pyrefly: ignore [missing-import]
import cv2 
import numpy as np
# pyrefly: ignore [missing-import]
import torch


class PostProcessor:
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert BCHW tensor (0-1) to uint8 OpenCV image.
        """

        image = (
            tensor.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)

        # RGB → BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    @staticmethod
    def save_image(image: np.ndarray, path: str):
        cv2.imwrite(path, image)