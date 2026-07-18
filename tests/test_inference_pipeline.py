"""
Integration test for the end-to-end inference pipeline.
Verifies:
  - The pipeline runs without exceptions.
  - The output interpolated PNG file is created successfully.
  - The output dimensions match the expected target dimensions (384x384).
"""

import os
from scripts.run_inference_smoke import run_pipeline


def test_inference_pipeline():
    # 1. Run the pipeline (which performs verification and returns the image path and shape)
    out_path, shape = run_pipeline()

    # 2. Assert output file is created and has a non-zero size
    assert os.path.exists(out_path), f"Output PNG not found at: {out_path}"
    assert os.path.getsize(out_path) > 0, f"Saved PNG at {out_path} is empty (0 bytes)"

    # 3. Assert dimensions match the inputs (resizing to 384x384 is contract in preprocessor)
    assert shape == (384, 384), f"Expected shape (384, 384), but got {shape}"

    print(
        f"Integration test passed successfully. Validated output at: {out_path} with shape {shape}"
    )
