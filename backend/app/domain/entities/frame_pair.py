from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from uuid import UUID, uuid4

from app.domain.entities.frame import Frame
from app.domain.entities.base_entity import BaseEntity
from app.shared.exceptions.domain import DomainValidationError


@dataclass(slots=True)
class FramePair(BaseEntity):
    """
    Represents two compatible frames for interpolation.
    """

    frame_a: Frame
    frame_b: Frame


    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the frame pair can be interpolated."""

        if self.frame_a.id == self.frame_b.id:
            raise DomainValidationError("FramePair cannot contain the same frame twice.")

        if self.frame_a.resolution != self.frame_b.resolution:
            raise DomainValidationError("Frame dimensions must match.")

        if (
            self.frame_a.metadata.channels
            != self.frame_b.metadata.channels
        ):
            raise DomainValidationError("Channel count must match.")

        if self.frame_a.provider != self.frame_b.provider:
            raise DomainValidationError("Provider mismatch.")

        if self.frame_a.dataset != self.frame_b.dataset:
            raise DomainValidationError("Dataset mismatch.")

        if self.frame_a.timestamp.value > self.frame_b.timestamp.value:
            raise DomainValidationError(
                "Frame A timestamp must be earlier than or equal to Frame B."
            )

    @property
    def dimensions(self):
        return self.frame_a.resolution

    @property
    def provider(self):
        return self.frame_a.provider

    @property
    def dataset(self):
        return self.frame_a.dataset

    @property
    def time_gap(self) -> timedelta:
        return (
            self.frame_b.timestamp.value
            - self.frame_a.timestamp.value
        )

    def swap(self) -> "FramePair":
        """
        Return a new FramePair with the frames swapped.
        """
        return FramePair(
            frame_a=self.frame_b,
            frame_b=self.frame_a,
        )

    def __str__(self) -> str:
        return (
            f"FramePair("
            f"{self.frame_a.path.name} -> "
            f"{self.frame_b.path.name})"
        )