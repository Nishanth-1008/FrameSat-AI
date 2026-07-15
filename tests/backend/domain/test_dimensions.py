# pyrefly: ignore [missing-import]
import pytest

# pyrefly: ignore [missing-import]
from app.domain.value_objects.dimensions import Dimensions
# pyrefly: ignore [missing-import]
from app.shared.exceptions.domain import DomainValidationError


class TestDimensions:
    """Tests for the Dimensions value object."""

    def test_creates_valid_dimensions(self):
        dims = Dimensions(512, 512)

        assert dims.width == 512
        assert dims.height == 512

    def test_computes_area(self):
        dims = Dimensions(512, 512)

        assert dims.area == 262144

    def test_computes_aspect_ratio(self):
        dims = Dimensions(400, 200)

        assert dims.aspect_ratio == 2.0

    def test_returns_tuple(self):
        dims = Dimensions(640, 480)

        assert dims.as_tuple() == (640, 480)

    def test_string_representation(self):
        dims = Dimensions(256, 128)

        assert str(dims) == "256 × 128"

    @pytest.mark.parametrize(
        "width,height",
        [
            (0, 100),
            (-1, 100),
            (100, 0),
            (100, -1),
            (0, 0),
        ],
    )
    def test_rejects_invalid_dimensions(self, width, height):
        with pytest.raises(DomainValidationError):
            Dimensions(width, height)