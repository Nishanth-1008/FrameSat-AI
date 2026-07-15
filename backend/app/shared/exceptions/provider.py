from .base import FrameSatError


class ProviderError(FrameSatError):
    """
    Raised when a provider fails.
    """

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(
            message,
            code="PROVIDER_ERROR",
        )
