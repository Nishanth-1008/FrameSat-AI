from __future__ import annotations

from dataclasses import dataclass

from app.shared.exceptions import DomainValidationError


@dataclass(frozen=True, slots=True)
class Dimensions:
    """
    Represents the dimensions of an image.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise DomainValidationError("Width must be greater than zero.")

        if self.height <= 0:
            raise DomainValidationError("Height must be greater than zero.")

    @property
    def area(self) -> int:
        """Returns the total number of pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Returns width divided by height."""
        return self.width / self.height

    def as_tuple(self) -> tuple[int, int]:
        """Returns (width, height)."""
        return (self.width, self.height)

    def __str__(self) -> str:
        return f"{self.width} × {self.height}"
