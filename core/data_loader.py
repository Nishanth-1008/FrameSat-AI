"""
Data Loader Module

Responsible for:
- Loading satellite images from disk (PNG, JPG, TIFF, NPY)
- Reading metadata (including CRS, transform, bounds for GeoTIFF)
- Returning NumPy arrays
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
import rasterio

from app.logger import logger
from core.exceptions import ValidationError
from core.constants import SUPPORTED_FORMATS

def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk and return a NumPy array.
    Supports PNG, JPG, JPEG, TIFF, GeoTIFF, and NPY.
    """
    path_obj = Path(path)
    logger.info(f"Loading Image: {path_obj}")

    if not path_obj.exists():
        logger.error(f"Validation Failed: File does not exist: {path_obj}")
        raise ValidationError(f"File does not exist: {path_obj}")

    ext = path_obj.suffix.lower()
    logger.info(f"Checking Format: {ext}")
    if ext not in SUPPORTED_FORMATS:
        logger.error(f"Validation Failed: Unsupported format {ext}")
        raise ValidationError(f"Unsupported image format: {ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}")

    try:
        if ext in (".png", ".jpg", ".jpeg"):
            with Image.open(path_obj) as PIL_img:
                img = PIL_img.convert("RGB")
                img_array = np.array(img)
        elif ext in (".tif", ".tiff"):
            with rasterio.open(path_obj) as src:
                img_array = src.read()
                # rasterio reads as (C, H, W), transpose to (H, W, C)
                img_array = np.moveaxis(img_array, 0, -1)
        elif ext == ".npy":
            img_array = np.load(path_obj)
        else:
            raise ValidationError(f"Unsupported format: {ext}")

        logger.info(f"Loaded {path_obj}")
        return img_array
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Validation Failed: Failed to open or read image {path_obj}: {str(e)}")
        raise ValidationError(f"Failed to load image: {path_obj}. The file may be corrupted or invalid.")


def load_pair(path_a: Union[str, Path], path_b: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load two images and return them as a tuple of NumPy arrays.
    """
    logger.info(f"Loading image pair: {path_a} and {path_b}")
    return load_image(path_a), load_image(path_b)


def read_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from the specified image file.
    Supports geotransforms for GeoTIFF images.
    """
    path_obj = Path(path)
    logger.info(f"Reading Metadata: {path_obj}")

    if not path_obj.exists():
        logger.error(f"Validation Failed: File does not exist: {path_obj}")
        raise ValidationError(f"File does not exist: {path_obj}")

    ext = path_obj.suffix.lower()
    filesize_mb = round(path_obj.stat().st_size / (1024 * 1024), 3)

    metadata = {
        "filename": path_obj.name,
        "format": ext.upper().lstrip("."),
        "width": 0,
        "height": 0,
        "channels": 0,
        "filesize_mb": filesize_mb,
        "crs": None,
        "transform": None,
        "bounds": None
    }

    try:
        if ext in (".png", ".jpg", ".jpeg"):
            with Image.open(path_obj) as PIL_img:
                metadata["width"], metadata["height"] = PIL_img.size
                if PIL_img.mode in ("RGB", "YCbCr", "LAB", "HSV"):
                    metadata["channels"] = 3
                elif PIL_img.mode in ("RGBA", "CMYK"):
                    metadata["channels"] = 4
                elif PIL_img.mode in ("L", "1", "P"):
                    metadata["channels"] = 1
                else:
                    metadata["channels"] = len(PIL_img.getbands())

        elif ext in (".tif", ".tiff"):
            with rasterio.open(path_obj) as src:
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["channels"] = src.count
                if src.crs:
                    metadata["crs"] = str(src.crs)
                if src.transform:
                    metadata["transform"] = list(src.transform) if hasattr(src.transform, "__iter__") else src.transform
                if src.bounds:
                    metadata["bounds"] = {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top
                    }

        elif ext == ".npy":
            arr = np.load(path_obj, mmap_mode="r")
            shape = arr.shape
            if len(shape) == 2:
                metadata["height"], metadata["width"] = shape
                metadata["channels"] = 1
            elif len(shape) == 3:
                metadata["height"], metadata["width"], metadata["channels"] = shape
            else:
                raise ValidationError(f"Invalid shape for NumPy array: {shape}")
        else:
            raise ValidationError(f"Unsupported format: {ext}")

        return metadata
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        logger.error(f"Failed to read metadata for {path_obj}: {str(e)}")
        raise ValidationError(f"Failed to read metadata: {path_obj}. The file may be corrupted or invalid.")


class DataLoader:
    """Wrapper class for backward compatibility."""
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        return load_image(image_path)

    @staticmethod
    def load_pair(frame1: str, frame2: str) -> Tuple[np.ndarray, np.ndarray]:
        return load_pair(frame1, frame2)

    @staticmethod
    def read_metadata(path: str) -> Dict[str, Any]:
        return read_metadata(path)