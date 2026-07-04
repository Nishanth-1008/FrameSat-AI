from functools import lru_cache
from pathlib import Path

from pydantic import Field
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application configuration.
    """

    APP_NAME: str = "FrameSat AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DEVICE: str = "cpu"

    OUTPUT_DIR: Path = Path("outputs")
    TEMP_DIR: Path = Path("temp")

    LOG_LEVEL: str = "INFO"

    MODEL_NAME: str = "RIFE"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()