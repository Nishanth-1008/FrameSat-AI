from .base import FrameSatError


class DomainValidationError(FrameSatError):
    """
    Raised when a domain object violates business rules.
    """

    def __init__(
        self,
        message: str,
    ) -> None:
        super().DomainValidationError(
            message,
            code="DOMAIN_VALIDATION_ERROR",
        )