from __future__ import annotations


class FrameSatError(Exception):
    """
    Base exception for the entire application.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "FRAMESAT_ERROR",
    ) -> None:
        self.message = message
        self.code = code
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
