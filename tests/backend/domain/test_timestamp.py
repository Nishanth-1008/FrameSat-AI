from datetime import datetime, timezone

# pyrefly: ignore [missing-import]
import pytest

from app.domain.value_objects.timestamp import Timestamp
from app.shared.exceptions.domain import DomainValidationError


class TestTimestamp:
    """Tests for the Timestamp value object."""

    def test_creates_valid_timestamp(self):
        dt = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

        ts = Timestamp(dt)

        assert ts.value == dt

    def test_now_returns_timestamp(self):
        ts = Timestamp.now()

        assert isinstance(ts, Timestamp)
        assert ts.value.tzinfo is not None

    def test_from_iso(self):
        iso = "2026-07-07T12:30:00+00:00"

        ts = Timestamp.from_iso(iso)

        assert ts.to_iso() == iso

    def test_to_iso(self):
        dt = datetime(
            2026,
            7,
            7,
            12,
            30,
            tzinfo=timezone.utc,
        )

        ts = Timestamp(dt)

        assert ts.to_iso() == dt.isoformat()

    def test_string_representation(self):
        ts = Timestamp.from_iso(
            "2026-07-07T12:30:00+00:00"
        )

        assert str(ts) == ts.to_iso()

    def test_invalid_type(self):
        with pytest.raises(DomainValidationError):
            Timestamp("2026-07-07")