from .base import FrameSatError


class ConfigurationError(FrameSatError):
    """
    Raised when application configuration is invalid.
    """

    def __init__(
        self,
        message: str,
    ) -> None:
        super().DomainValidationError(
            message,
            code="CONFIGURATION_ERROR",
        )