import unittest
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image

from core.data_loader import load_image, load_pair, read_metadata
from core.exceptions import ValidationError

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "temp_test_data"
        self.test_dir.mkdir(exist_ok=True)

        # Create valid PNG (512x512, RGB)
        self.png_path = self.test_dir / "valid.png"
        img = Image.new("RGB", (512, 512), color="red")
        img.save(self.png_path)

        # Create valid NPY (512x512, 3 channels)
        self.npy_path = self.test_dir / "valid.npy"
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        np.save(self.npy_path, arr)

    def tearDown(self):
        # Cleanup
        for f in self.test_dir.iterdir():
            f.unlink()
        self.test_dir.rmdir()

    def test_load_png(self):
        img = load_image(self.png_path)
        self.assertEqual(img.shape, (512, 512, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_load_npy(self):
        img = load_image(self.npy_path)
        self.assertEqual(img.shape, (512, 512, 3))

    def test_load_pair(self):
        img_a, img_b = load_pair(self.png_path, self.npy_path)
        self.assertEqual(img_a.shape, (512, 512, 3))
        self.assertEqual(img_b.shape, (512, 512, 3))

    def test_nonexistent_file(self):
        with self.assertRaises(ValidationError):
            load_image(self.test_dir / "does_not_exist.png")

    def test_unsupported_format(self):
        txt_path = self.test_dir / "test.txt"
        with open(txt_path, "w") as f:
            f.write("hello")
        with self.assertRaises(ValidationError):
            load_image(txt_path)

    def test_read_metadata_png(self):
        meta = read_metadata(self.png_path)
        self.assertEqual(meta["filename"], "valid.png")
        self.assertEqual(meta["format"], "PNG")
        self.assertEqual(meta["width"], 512)
        self.assertEqual(meta["height"], 512)
        self.assertEqual(meta["channels"], 3)
        self.assertGreater(meta["filesize_mb"], 0)

if __name__ == "__main__":
    unittest.main()
