"""
End-to-end compatibility test: SEVIR frames -> existing core pipeline.

This proves SEVIRProvider's output (HxWx3 uint8 images) satisfies the same
contract as ImageProvider's uploaded-file output, so the two data sources
can share one interpolation pipeline (core/validator.py, preprocessor.py,
tensor_converter.py) exactly as described in the research report.
"""

import numpy as np

from core.preprocessor import ImagePreprocessor
from core.tensor_converter import TensorConverter
from core.validator import ImageValidator


def test_sevir_frame_pair_passes_image_validator(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=4, frame_b=6)

    # Should not raise -- same validation uploaded images go through.
    ImageValidator.validate(pair.image_a, pair.image_b)


def test_sevir_frame_pair_preprocesses_to_normalized_float(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=4, frame_b=6)

    processed_a = ImagePreprocessor.preprocess(pair.image_a)
    processed_b = ImagePreprocessor.preprocess(pair.image_b)

    for arr in (processed_a, processed_b):
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0
        assert arr.shape[-1] == 3


def test_sevir_frame_converts_to_bchw_tensor(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=4, frame_b=6)

    processed = ImagePreprocessor.preprocess(pair.image_a)
    tensor = TensorConverter.to_tensor(processed)

    assert tensor.shape[0] == 1  # batch
    assert tensor.shape[1] == 3  # channels
    assert tensor.shape[2] == pair.image_a.shape[0]
    assert tensor.shape[3] == pair.image_a.shape[1]


def test_ground_truth_frame_also_pipeline_compatible(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=4, frame_b=6)

    assert pair.ground_truth_image is not None
    # Ground truth must be directly comparable (same shape/dtype) to a
    # model output that's gone through the same pipeline as image_a/image_b.
    assert pair.ground_truth_image.shape == pair.image_a.shape
    assert pair.ground_truth_image.dtype == pair.image_a.dtype
