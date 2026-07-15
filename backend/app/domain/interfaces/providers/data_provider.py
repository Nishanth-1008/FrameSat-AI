from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.frame import Frame
from app.domain.entities.frame_pair import FramePair
from app.domain.entities.frame_sequence import FrameSequence
from app.domain.value_objects.metadata import Metadata


class DataProvider(ABC):
    """
    Base contract implemented by every dataset provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Provider version."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize provider resources."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release provider resources."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate provider configuration."""

    @abstractmethod
    def load_frame(self, identifier: str) -> Frame:
        """Load a single frame."""

    @abstractmethod
    def load_pair(
        self,
        first: str,
        second: str,
    ) -> FramePair:
        """Load two frames."""

    @abstractmethod
    def load_sequence(
        self,
        identifiers: list[str],
    ) -> FrameSequence:
        """Load multiple frames."""

    @abstractmethod
    def metadata(self, identifier: str) -> Metadata:
        """Return metadata without loading image pixels."""
