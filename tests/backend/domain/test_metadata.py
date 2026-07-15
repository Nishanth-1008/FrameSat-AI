from datetime import datetime, timezone

import pytest

from app.domain.enums.dataset_type import DatasetType
from app.domain.enums.provider_type import ProviderType
from app.domain.value_objects.dimensions import Dimensions
from app.domain.value_objects.metadata import Metadata
from app.domain.value_objects.timestamp import Timestamp
from app.shared.exceptions import DomainValidationError


class TestMetadata:
    """Tests for the Metadata value object."""

    def test_creates_valid_metadata(self):
        metadata = Metadata(
            provider=ProviderType.UPLOAD,
            dataset=DatasetType.USER_UPLOAD,
            timestamp=Timestamp(
                datetime.now(timezone.utc)
            ),
            dimensions=Dimensions(512, 512),
            channels=3,
        )

        assert metadata.provider == ProviderType.UPLOAD
        assert metadata.dataset == DatasetType.USER_UPLOAD
        assert metadata.channels == 3

    def test_default_channel_count(self):
        metadata = Metadata(
            provider=ProviderType.UPLOAD,
            dataset=DatasetType.USER_UPLOAD,
            timestamp=Timestamp.now(),
            dimensions=Dimensions(512, 512),
        )

        assert metadata.channels == 3

    @pytest.mark.parametrize(
        "channels",
        [
            0,
            -1,
            -5,
        ],
    )
    def test_rejects_invalid_channels(self, channels):
        with pytest.raises(DomainValidationError):
            Metadata(
                provider=ProviderType.UPLOAD,
                dataset=DatasetType.USER_UPLOAD,
                timestamp=Timestamp.now(),
                dimensions=Dimensions(512, 512),
                channels=channels,
            )

    def test_optional_fields(self):
        metadata = Metadata(
            provider=ProviderType.UPLOAD,
            dataset=DatasetType.USER_UPLOAD,
            timestamp=Timestamp.now(),
            dimensions=Dimensions(256, 256),
            satellite="INSAT-3D",
            sensor="IMAGER",
        )

        assert metadata.satellite == "INSAT-3D"
        assert metadata.sensor == "IMAGER"

    def test_empty_provider_not_allowed(self):
        with pytest.raises(Exception):
            Metadata(
                provider="",  # type: ignore[arg-type]
                dataset=DatasetType.USER_UPLOAD,
                timestamp=Timestamp.now(),
                dimensions=Dimensions(512, 512),
            )

    def test_empty_dataset_not_allowed(self):
        with pytest.raises(Exception):
            Metadata(
                provider=ProviderType.UPLOAD,
                dataset="",  # type: ignore[arg-type]
                timestamp=Timestamp.now(),
                dimensions=Dimensions(512, 512),
            )