"""
Image Validation Module

Validates image formats, shapes, file sizes, corruption, and channel compatibilities.
"""

from pathlib import Path
from typing import Union
import numpy as np

from app.logger import logger
from app.config import MAX_UPLOAD_SIZE_MB
from core.constants import SUPPORTED_FORMATS
from core.exceptions import ValidationError
from core.data_loader import load_image

def validate_image(path: Union[str, Path]) -> np.ndarray:
    """
    Validate a single image file for existence, extension support, size limit,
    and corruption. Returns the loaded array if valid.
    """
    path_obj = Path(path)
    logger.info(f"Validating Image: {path_obj}")

    if not path_obj.exists():
        logger.error(f"Validation Failed: File does not exist: {path_obj}")
        raise ValidationError(f"File does not exist: {path_obj}")

    # Check extension
    ext = path_obj.suffix.lower()
    logger.info("Checking Format")
    if ext not in SUPPORTED_FORMATS:
        logger.error(f"Validation Failed: Unsupported format {ext}")
        raise ValidationError(
            "Unsupported image format.\n"
            "Supported:\n"
            "PNG\n"
            "JPG\n"
            "TIFF\n"
            "NPY"
        )

    # Check size limit
    file_size_mb = path_obj.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        logger.error(f"Validation Failed: File size {file_size_mb:.2f} MB exceeds limit {MAX_UPLOAD_SIZE_MB} MB")
        raise ValidationError(f"File size {file_size_mb:.2f} MB exceeds maximum limit of {MAX_UPLOAD_SIZE_MB} MB.")

    # Check corruption and load
    try:
        img = load_image(path_obj)
    except Exception as e:
        logger.error(f"Validation Failed: File corrupted or unreadable: {str(e)}")
        raise ValidationError(f"File is corrupted or could not be read: {path_obj.name}")

    # Check dimensions
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        logger.error(f"Validation Failed: Dimensions are zero: {w}x{h}")
        raise ValidationError(f"Image has invalid dimensions: {w}x{h}")

    # Check channels
    channels = img.shape[2] if len(img.shape) == 3 else 1
    if channels not in (1, 3, 4):
        logger.error(f"Validation Failed: Unsupported channel count {channels}")
        raise ValidationError(f"Image has invalid number of channels: {channels}. Supported: 1, 3, or 4 channels.")

    return img


def validate_pair(path_a: Union[str, Path], path_b: Union[str, Path]) -> bool:
    """
    Validate a pair of images for size compatibility and channels alignment.
    """
    logger.info(f"Validating Image Pair: {path_a} and {path_b}")

    img_a = validate_image(path_a)
    img_b = validate_image(path_b)

    # Check same dimensions
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    if (h_a, w_a) != (h_b, w_b):
        logger.error(f"Validation Failed: Image dimensions do not match: {w_a}x{h_a} != {w_b}x{h_b}")
        raise ValidationError(
            f"Image dimensions do not match.\n"
            f"Frame A\n"
            f"{w_a}x{h_a}\n"
            f"Frame B\n"
            f"{w_b}x{h_b}"
        )

    # Check channels
    c_a = img_a.shape[2] if len(img_a.shape) == 3 else 1
    c_b = img_b.shape[2] if len(img_b.shape) == 3 else 1
    if c_a != c_b:
        logger.error(f"Validation Failed: Channel counts do not match: {c_a} != {c_b}")
        raise ValidationError(f"Image channel count does not match: {c_a} != {c_b}")

    logger.info("Validation Passed")
    return True


class ImageValidator:
    """Wrapper class for backward compatibility."""
    @staticmethod
    def validate(img1: np.ndarray, img2: np.ndarray) -> None:
        """Validate loaded arrays compatibility."""
        if img1 is None or img2 is None:
            raise ValidationError("One or both images are None.")

        if img1.shape != img2.shape:
            h_a, w_a = img1.shape[:2]
            h_b, w_b = img2.shape[:2]
            raise ValidationError(
                f"Image dimensions do not match.\n"
                f"Frame A\n"
                f"{w_a}x{h_a}\n"
                f"Frame B\n"
                f"{w_b}x{h_b}"
            )

        if len(img1.shape) not in (2, 3):
            raise ValidationError("Images must have 2 or 3 dimensions.")

        channels = img1.shape[2] if len(img1.shape) == 3 else 1
        if channels not in (1, 3, 4):
            raise ValidationError(f"Image has invalid number of channels: {channels}. Supported: 1, 3, or 4 channels.")