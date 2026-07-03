import unittest
import numpy as np
from pathlib import Path
from PIL import Image

from core.validator import validate_image, validate_pair
from core.exceptions import ValidationError

class TestValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "temp_test_validator"
        self.test_dir.mkdir(exist_ok=True)

        # Create valid PNG (512x512, RGB)
        self.valid_png = self.test_dir / "valid.png"
        img = Image.new("RGB", (512, 512), color="blue")
        img.save(self.valid_png)

        # Create another valid PNG with same size
        self.valid_png2 = self.test_dir / "valid2.png"
        img2 = Image.new("RGB", (512, 512), color="green")
        img2.save(self.valid_png2)

        # Create PNG with different size (256x256)
        self.diff_size_png = self.test_dir / "diff_size.png"
        img_diff = Image.new("RGB", (256, 256), color="red")
        img_diff.save(self.diff_size_png)

        # Create invalid extension file
        self.invalid_ext = self.test_dir / "invalid.txt"
        with open(self.invalid_ext, "w") as f:
            f.write("text content")

        # Create corrupted image (zero bytes or invalid contents with image ext)
        self.corrupted_png = self.test_dir / "corrupted.png"
        with open(self.corrupted_png, "wb") as f:
            f.write(b"NOT_A_PNG_FILE_AT_ALL")

        # Create too large file
        self.large_file = self.test_dir / "large.png"
        with open(self.large_file, "wb") as f:
            f.seek(26 * 1024 * 1024 - 1)
            f.write(b"\0")

    def tearDown(self):
        # Cleanup
        for f in self.test_dir.iterdir():
            f.unlink()
        self.test_dir.rmdir()

    def test_valid_image(self):
        img = validate_image(self.valid_png)
        self.assertEqual(img.shape, (512, 512, 3))

    def test_invalid_extension(self):
        with self.assertRaises(ValidationError) as context:
            validate_image(self.invalid_ext)
        self.assertIn("Unsupported image format", str(context.exception))

    def test_corrupted_image(self):
        with self.assertRaises(ValidationError) as context:
            validate_image(self.corrupted_png)
        self.assertIn("corrupted or could not be read", str(context.exception))

    def test_too_large_file(self):
        with self.assertRaises(ValidationError) as context:
            validate_image(self.large_file)
        self.assertIn("exceeds maximum limit", str(context.exception))

    def test_validate_pair_success(self):
        self.assertTrue(validate_pair(self.valid_png, self.valid_png2))

    def test_validate_pair_dimension_mismatch(self):
        with self.assertRaises(ValidationError) as context:
            validate_pair(self.valid_png, self.diff_size_png)
        self.assertIn("dimensions do not match", str(context.exception))

if __name__ == "__main__":
    unittest.main()
