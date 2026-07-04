from __future__ import annotations

from dataclasses import dataclass

from .dimensions import Dimensions
from .timestamp import Timestamp
from app.domain.enums.provider_type import ProviderType
from app.domain.enums.dataset_type import DatasetType
from app.shared.exceptions.domain import DomainValidationError

@dataclass(frozen=True, slots=True)
class Metadata:
    """
    Standardized metadata shared across all supported datasets.

    Every provider converts its native metadata into this object before
    passing data to the rest of the application.
    """

    provider: ProviderType
    dataset: DatasetType
    timestamp: Timestamp
    dimensions: Dimensions

    channels: int = 3

    sensor: str | None = None
    satellite: str | None = None

    spatial_resolution: float | None = None

    event_id: str | None = None
    sample_id: str | None = None

    projection: str | None = None

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise DomainValidationError("Channels must be greater than zero.")

        if not self.provider.strip():
            raise DomainValidationError("Provider cannot be empty.")

        if not self.dataset.strip():
            raise DomainValidationError("Dataset cannot be empty.")