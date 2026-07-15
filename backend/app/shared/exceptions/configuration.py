from .base import FrameSatError


class ConfigurationError(FrameSatError):
    """
    Raised when application configuration is invalid.
    """

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(
            message,
            code="CONFIGURATION_ERROR",
        )
