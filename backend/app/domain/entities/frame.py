from __future__ import annotations

from app.domain.entities.base_entity import BaseEntity

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from app.domain.value_objects.metadata import Metadata
from app.domain.enums.provider_type import ProviderType
from app.domain.enums.dataset_type import DatasetType
from app.domain.value_objects.timestamp import Timestamp
from app.domain.value_objects.dimensions import Dimensions

@dataclass(slots=True)
class Frame(BaseEntity):
    """
    Represents a single satellite observation.

    A Frame is a domain entity that references an image on disk
    together with its standardized metadata.
    """

    path: Path
    metadata: Metadata

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def width(self) -> int:
        return self.metadata.dimensions.width

    @property
    def height(self) -> int:
        return self.metadata.dimensions.height

    @property
    def provider(self) -> ProviderType:
        return self.metadata.provider

    @property
    def dataset(self) -> DatasetType:
        return self.metadata.dataset

    @property
    def timestamp(self) -> Timestamp:
        return self.metadata.timestamp

    @property
    def resolution(self) -> Dimensions:
        return self.metadata.dimensions

    def exists(self) -> bool:
        """Returns True if the image exists."""
        return self.path.exists()

    def __str__(self) -> str:
        return (
            f"Frame("
            f"id={self.id}, "
            f"path='{self.path.name}', "
            f"provider={self.provider.value}, "
            f"dataset={self.dataset.value})"
        )