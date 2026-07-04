import pandas as pd
import pytest

from app.sevir.catalog import (
    CatalogError,
    frame_offsets_minutes,
    get_event_row_for_type,
    get_event_rows,
    list_events,
    load_catalog,
)


def test_load_catalog_success(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    assert not df.empty
    assert set(["id", "file_name", "file_index", "img_type", "time_utc", "minute_offsets"]).issubset(
        df.columns
    )


def test_load_catalog_missing_file(tmp_path):
    with pytest.raises(CatalogError, match="not found"):
        load_catalog(tmp_path / "does_not_exist.csv")


def test_load_catalog_missing_required_column(tmp_path):
    bad_csv = tmp_path / "CATALOG.csv"
    pd.DataFrame({"id": ["a"], "img_type": ["vil"]}).to_csv(bad_csv, index=False)
    with pytest.raises(CatalogError, match="missing required columns"):
        load_catalog(bad_csv)


def test_load_catalog_unknown_img_type(tmp_path):
    bad_csv = tmp_path / "CATALOG.csv"
    pd.DataFrame(
        {
            "id": ["a"],
            "file_name": ["f.h5"],
            "file_index": [0],
            "img_type": ["not_a_real_type"],
            "time_utc": ["2020-01-01"],
            "minute_offsets": ["0"],
        }
    ).to_csv(bad_csv, index=False)
    with pytest.raises(CatalogError, match="unrecognized img_type"):
        load_catalog(bad_csv)


def test_list_events_groups_by_id(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    events = list_events(df)
    # fixture generates 5 events, each covered by all 4 raster types
    assert len(events) == 5
    for e in events:
        assert set(e.img_types) == {"vis", "ir069", "ir107", "vil"}


def test_list_events_filters_by_img_type_subset(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    events = list_events(df, img_types=("vil", "vis"))
    assert len(events) == 5  # all events have both in this fixture

    events_missing_type = list_events(df, img_types=("vil", "nonexistent"))
    assert events_missing_type == []


def test_list_events_filters_by_year(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    events_2019 = list_events(df, year=2019)
    events_2099 = list_events(df, year=2099)
    assert len(events_2019) == 5
    assert events_2099 == []


def test_get_event_rows_unknown_id_raises(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    with pytest.raises(CatalogError, match="No SEVIR event found"):
        get_event_rows(df, "does-not-exist")


def test_get_event_row_for_type(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    events = list_events(df)
    row = get_event_row_for_type(df, events[0].event_id, "vil")
    assert row["img_type"] == "vil"


def test_get_event_row_for_type_missing_type_raises(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    events = list_events(df)
    with pytest.raises(CatalogError, match="no img_type"):
        get_event_row_for_type(df, events[0].event_id, "lght")


def test_frame_offsets_minutes_parses_colon_separated(sevir_fixture_root):
    _, catalog_path = sevir_fixture_root
    df = load_catalog(catalog_path)
    events = list_events(df)
    row = get_event_row_for_type(df, events[0].event_id, "vil")
    offsets = frame_offsets_minutes(row)
    assert len(offsets) == 49
    assert offsets[0] == -120
    assert offsets[-1] == 120
    # 5-minute cadence
    assert offsets[1] - offsets[0] == 5
