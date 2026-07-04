import numpy as np
import pytest

from app.sevir.provider import SEVIRProviderError


def test_list_events_returns_fixture_events(sevir_provider):
    events = sevir_provider.list_events()
    assert len(events) == 5


def test_list_events_filter_by_img_type(sevir_provider):
    events = sevir_provider.list_events(img_types=("vil",))
    assert len(events) == 5


def test_get_event_summary_unknown_id_raises(sevir_provider):
    with pytest.raises(SEVIRProviderError, match="Unknown SEVIR event"):
        sevir_provider.get_event_summary("nope")


def test_list_frames_returns_49_frames(sevir_provider):
    events = sevir_provider.list_events()
    frames = sevir_provider.list_frames(events[0].event_id, "vil")
    assert len(frames) == 49
    assert frames[0].offset_minutes == -120
    assert frames[-1].offset_minutes == 120


def test_get_frame_image_shape_and_dtype(sevir_provider):
    events = sevir_provider.list_events()
    img = sevir_provider.get_frame_image(events[0].event_id, "vil", 10)
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8


def test_get_frame_image_out_of_range_raises(sevir_provider):
    events = sevir_provider.list_events()
    with pytest.raises(SEVIRProviderError, match="out of range"):
        sevir_provider.get_frame_image(events[0].event_id, "vil", 999)


def test_get_frame_pair_basic(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=10, frame_b=12)
    assert pair.image_a.shape == (32, 32, 3)
    assert pair.image_b.shape == (32, 32, 3)
    assert pair.frame_a_index == 10
    assert pair.frame_b_index == 12


def test_get_frame_pair_provides_ground_truth_when_adjacent_gap(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=4, frame_b=6)
    assert pair.ground_truth_index == 5
    assert pair.ground_truth_image is not None
    assert pair.ground_truth_image.shape == (32, 32, 3)


def test_get_frame_pair_no_ground_truth_when_gap_not_two(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=0, frame_b=10)
    assert pair.ground_truth_index is None
    assert pair.ground_truth_image is None


def test_get_frame_pair_can_disable_ground_truth(sevir_provider):
    events = sevir_provider.list_events()
    pair = sevir_provider.get_frame_pair(
        events[0].event_id, "vil", frame_a=4, frame_b=6, include_ground_truth=False
    )
    assert pair.ground_truth_index is None


def test_get_frame_pair_rejects_equal_indices(sevir_provider):
    events = sevir_provider.list_events()
    with pytest.raises(SEVIRProviderError, match="must differ"):
        sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=5, frame_b=5)


def test_get_frame_pair_rejects_out_of_range(sevir_provider):
    events = sevir_provider.list_events()
    with pytest.raises(SEVIRProviderError, match="out of range"):
        sevir_provider.get_frame_pair(events[0].event_id, "vil", frame_a=0, frame_b=999)


def test_missing_data_dir_raises_helpful_error(sevir_fixture_root):
    from app.sevir.provider import SEVIRProvider

    root, catalog_path = sevir_fixture_root
    provider = SEVIRProvider(catalog_path=catalog_path, data_dir=root / "nonexistent_dir")
    events = provider.list_events()
    with pytest.raises(SEVIRProviderError, match="not found"):
        provider.get_frame_image(events[0].event_id, "vil", 0)
