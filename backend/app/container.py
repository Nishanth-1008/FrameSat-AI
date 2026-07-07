from __future__ import annotations

from app.shared.config.settings import get_settings
from app.infrastructure.logging import get_logger


class ApplicationContainer:
    """
    Composition root for the application.

    Responsible for creating and exposing shared services.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger("FrameSatAI")


container = ApplicationContainer()