from enum import Enum


class ImageFormat(str, Enum):
    """Supported image formats."""

    PNG = "png"

    JPG = "jpg"

    JPEG = "jpeg"

    TIFF = "tiff"

    GEOTIFF = "geotiff"

    NUMPY = "npy"
