from enum import Enum


class ModelType(str, Enum):
    """Supported interpolation models."""

    RIFE = "rife"

    FILM = "film"

    IFRNET = "ifrnet"

    AMT = "amt"
