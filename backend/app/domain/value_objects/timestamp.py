from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from app.shared.exceptions.domain import DomainValidationError

@dataclass(frozen=True, slots=True)
class Timestamp:
    """
    Represents a point in time associated with a satellite observation.
    """

    value: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.value, datetime):
            raise DomainValidationError("value must be a datetime instance.")

    @classmethod
    def now(cls) -> "Timestamp":
        """Create a Timestamp using the current UTC time."""
        return cls(datetime.now(timezone.utc))

    @classmethod
    def from_iso(cls, value: str) -> "Timestamp":
        """Create a Timestamp from an ISO-8601 string."""
        return cls(datetime.fromisoformat(value))

    def to_iso(self) -> str:
        """Return the timestamp as an ISO-8601 string."""
        return self.value.isoformat()

    def __str__(self) -> str:
        return self.to_iso()