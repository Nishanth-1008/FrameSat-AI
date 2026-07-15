from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Configuration module for the FrameSat AI application.

    Loads configuration settings from environment variables and/or a local .env file.
    Provides type-safe config validation and sensible defaults for local development.
    """

    MODEL_WEIGHTS_PATH: Path = Field(
        default=Path("./weights/rife.pth"),
        description="Path to the model weights file (e.g. RIFE checkpoint).",
    )

    INPUT_DIR: Path = Field(
        default=Path("./assets/sample_inputs"),
        description="Directory containing input images or datasets.",
    )

    OUTPUT_DIR: Path = Field(
        default=Path("./assets/sample_outputs"),
        description="Directory where outputs and interpolation results will be saved.",
    )

    DEVICE: str = Field(
        default="cpu",
        description="Computation device to use for inference (e.g. 'cpu', 'cuda', 'mps').",
    )

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level for the application."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate the configuration settings object
_config = AppConfig()

# Expose fields at the module level as requested
MODEL_WEIGHTS_PATH: Path = _config.MODEL_WEIGHTS_PATH
INPUT_DIR: Path = _config.INPUT_DIR
OUTPUT_DIR: Path = _config.OUTPUT_DIR
DEVICE: str = _config.DEVICE
LOG_LEVEL: str = _config.LOG_LEVEL
