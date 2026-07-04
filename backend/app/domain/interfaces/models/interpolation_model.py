from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.frame_pair import FramePair
from app.domain.entities.frame import Frame
from app.domain.enums.device_type import DeviceType


class InterpolationModel(ABC):
    """
    Contract implemented by every interpolation model.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version."""

    @property
    @abstractmethod
    def device(self) -> DeviceType:
        """Execution device."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def warmup(self) -> None:
        """Warm the model for inference."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abstractmethod
    def interpolate(self, pair: FramePair) -> Frame:
        """
        Generate an intermediate frame.

        Returns
        -------
        Frame
            Generated frame.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether the model is ready."""