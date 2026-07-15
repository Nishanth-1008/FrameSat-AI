from enum import Enum


class ProviderType(str, Enum):
    """Supported data providers."""

    UPLOAD = "upload"
    SEVIR = "sevir"
    NOAA = "noaa"
    INSAT = "insat"
    SENTINEL = "sentinel"
