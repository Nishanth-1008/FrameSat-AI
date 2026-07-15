from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True)
class BaseEntity:
    """
    Base class for all domain entities.

    Every entity has a unique identity that remains constant
    throughout its lifetime.
    """

    id: UUID = field(default_factory=uuid4, kw_only=True)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEntity):
            return False

        return self.id == other.id
