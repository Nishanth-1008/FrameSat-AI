from functools import lru_cache
from pathlib import Path

# pyrefly: ignore [missing-import]
from pydantic_settings import SettingsConfigDict
from app.config import AppConfig


class Settings(AppConfig):
    """
    Global application configuration extending the core AppConfig.
    """

    APP_NAME: str = "FrameSat AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    TEMP_DIR: Path = Path("temp")
    MODEL_NAME: str = "RIFE"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()
