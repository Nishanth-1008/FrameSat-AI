from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from uuid import UUID, uuid4

from app.domain.entities.frame import Frame
from app.domain.entities.frame_pair import FramePair
from app.domain.entities.base_entity import BaseEntity
from app.shared.exceptions.domain import DomainValidationError

@dataclass(slots=True)
class FrameSequence(BaseEntity):
    """
    Represents an ordered temporal sequence of satellite frames.
    """

    frames: list[Frame]


    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if len(self.frames) < 2:
            raise DomainValidationError(
                "A FrameSequence must contain at least two frames."
            )

        first = self.frames[0]

        for previous, current in zip(self.frames, self.frames[1:]):

            if previous.timestamp.value > current.timestamp.value:
                raise DomainValidationError(
                    "Frames must be ordered chronologically."
                )

        for frame in self.frames:

            if frame.provider != first.provider:
                raise DomainValidationError("Provider mismatch.")

            if frame.dataset != first.dataset:
                raise DomainValidationError("Dataset mismatch.")

            if frame.resolution != first.resolution:
                raise DomainValidationError("Resolution mismatch.")

            if (
                frame.metadata.channels
                != first.metadata.channels
            ):
                raise DomainValidationError("Channel mismatch.")

    @property
    def provider(self):
        return self.frames[0].provider

    @property
    def dataset(self):
        return self.frames[0].dataset

    @property
    def resolution(self):
        return self.frames[0].resolution

    @property
    def start_time(self):
        return self.frames[0].timestamp

    @property
    def end_time(self):
        return self.frames[-1].timestamp

    @property
    def duration(self) -> timedelta:
        return (
            self.end_time.value
            - self.start_time.value
        )

    @property
    def length(self) -> int:
        return len(self.frames)

    def first(self) -> Frame:
        return self.frames[0]

    def last(self) -> Frame:
        return self.frames[-1]

    def middle(self) -> list[Frame]:
        return self.frames[1:-1]

    def pairwise(self) -> list[FramePair]:
        """
        Returns consecutive FramePairs.

        Example:

        A B C D

        →

        AB
        BC
        CD
        """
        return [
            FramePair(a, b)
            for a, b in zip(self.frames, self.frames[1:])
        ]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)