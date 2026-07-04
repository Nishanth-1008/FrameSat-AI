"""
SEVIR Catalog Module

Loads and queries `CATALOG.csv`, the metadata index that ties SEVIR event
IDs to the HDF5 files/rows containing their image data.

Schema reference (SEVIR_Tutorial.ipynb, Table 2):

    id              Unique event ID. Up to 5 rows share an `id` (one per
                    img_type that covers the event).
    file_name       HDF5 file containing the image data for this row.
    file_index      Row index within `file_name` where the event's data is.
    img_type        One of vis | ir069 | ir107 | vil | lght.
    time_utc        UTC timestamp, typically the middle frame of the event.
    minute_offsets  Colon-separated per-frame offsets (minutes) from time_utc.
    episode_id      NWS Storm Event EPISODE_ID (optional).
    event_id        NWS Storm Event EVENT_ID (optional).
    llcrnrlat/lon   Lower-left corner lat/lon.
    urcrnrlat/lon   Upper-right corner lat/lon.
    proj            Proj4 string (LAEA projection).
    size_x/size_y   Image size in pixels.
    height_m/width_m  Patch size in meters.
    data_min/data_max Min/max data value across the event's frames.
    pct_missing     Percentage of missing values.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import SEVIR_CATALOG_PATH, SEVIR_ALL_TYPES

# Columns we require to be present for the catalog to be usable. Some
# optional columns (episode_id/event_id, georeferencing) may be sparsely
# populated (NaN) for events not tied to an NWS Storm Event, so we don't
# require every catalog column -- just the ones the app depends on directly.
REQUIRED_COLUMNS = (
    "id",
    "file_name",
    "file_index",
    "img_type",
    "time_utc",
    "minute_offsets",
)


class CatalogError(Exception):
    """Raised when the SEVIR catalog is missing, malformed, or unreadable."""


@dataclass(frozen=True)
class EventSummary:
    """Lightweight summary of one SEVIR event, for list/browse views."""

    event_id: str
    time_utc: pd.Timestamp
    img_types: tuple[str, ...]
    episode_id: str | None
    nws_event_id: str | None
    llcrnrlat: float | None
    llcrnrlon: float | None
    urcrnrlat: float | None
    urcrnrlon: float | None

    @property
    def year(self) -> int:
        return int(self.time_utc.year)

    @property
    def display_name(self) -> str:
        loc = ""
        if self.llcrnrlat is not None and self.llcrnrlon is not None:
            loc = f" near ({self.llcrnrlat:.1f}, {self.llcrnrlon:.1f})"
        return f"Storm Event {self.event_id}{loc}"

    def to_dict(self) -> dict:
        def _native(v):
            """Cast numpy scalar types (int64/float64) to native Python types."""
            if v is None:
                return None
            if hasattr(v, "item"):
                return v.item()
            return v

        return {
            "event_id": self.event_id,
            "name": self.display_name,
            "date": self.time_utc.date().isoformat(),
            "time_utc": self.time_utc.isoformat(),
            "img_types": list(self.img_types),
            "episode_id": _native(self.episode_id),
            "nws_event_id": _native(self.nws_event_id),
            "bbox": (
                {
                    "llcrnrlat": _native(self.llcrnrlat),
                    "llcrnrlon": _native(self.llcrnrlon),
                    "urcrnrlat": _native(self.urcrnrlat),
                    "urcrnrlon": _native(self.urcrnrlon),
                }
                if self.llcrnrlat is not None
                else None
            ),
        }


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CatalogError(
            f"SEVIR catalog at {path} is missing required columns: {missing}. "
            "Verify it matches the CATALOG.csv schema described in the SEVIR "
            "tutorial (Table 2)."
        )


def load_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and lightly validate CATALOG.csv into a DataFrame.

    Raises CatalogError with a clear message if the file is absent or
    doesn't match the expected schema, rather than letting a raw pandas
    exception bubble up.
    """
    csv_path = Path(path) if path is not None else SEVIR_CATALOG_PATH

    if not csv_path.exists():
        raise CatalogError(
            f"SEVIR catalog not found at {csv_path}. Set SEVIR_CATALOG_PATH "
            "(or SEVIR_ROOT) to point at a local CATALOG.csv, e.g. downloaded "
            "via `aws s3 cp --no-sign-request s3://sevir/CATALOG.csv .`"
        )

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise CatalogError(f"Failed to parse SEVIR catalog at {csv_path}: {exc}") from exc

    _validate_columns(df, csv_path)

    try:
        df["time_utc"] = pd.to_datetime(df["time_utc"])
    except Exception as exc:  # pragma: no cover - defensive
        raise CatalogError(
            f"SEVIR catalog at {csv_path} has an unparseable time_utc column: {exc}"
        ) from exc

    unknown_types = set(df["img_type"].unique()) - set(SEVIR_ALL_TYPES)
    if unknown_types:
        raise CatalogError(
            f"SEVIR catalog at {csv_path} contains unrecognized img_type "
            f"values: {sorted(unknown_types)}. Expected one of {SEVIR_ALL_TYPES}."
        )

    return df


@lru_cache(maxsize=4)
def _load_catalog_cached(path_str: str) -> pd.DataFrame:
    return load_catalog(path_str)


def get_catalog(path: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load the catalog, cached by path (process-lifetime cache)."""
    csv_path = Path(path) if path is not None else SEVIR_CATALOG_PATH
    if not use_cache:
        return load_catalog(csv_path)
    return _load_catalog_cached(str(csv_path))


def clear_catalog_cache() -> None:
    _load_catalog_cached.cache_clear()


def _row_optional(row: pd.Series, col: str):
    if col not in row or pd.isna(row[col]):
        return None
    return row[col]


def list_events(
    df: pd.DataFrame,
    img_types: tuple[str, ...] | None = None,
    year: int | None = None,
) -> list[EventSummary]:
    """
    Group catalog rows by event `id` and return one EventSummary per event.

    Args:
        df: catalog DataFrame (from get_catalog()).
        img_types: if given, only include events that have ALL of these
            image types available (mirrors the tutorial's Example 2 filter).
        year: if given, only include events whose time_utc falls in this year.
    """
    grouped = df.groupby("id")

    summaries: list[EventSummary] = []
    for event_id, group in grouped:
        available_types = tuple(sorted(group["img_type"].unique()))

        if img_types is not None and not set(img_types).issubset(set(available_types)):
            continue

        first = group.iloc[0]
        time_utc = pd.Timestamp(first["time_utc"])

        if year is not None and time_utc.year != year:
            continue

        summaries.append(
            EventSummary(
                event_id=str(event_id),
                time_utc=time_utc,
                img_types=available_types,
                episode_id=_row_optional(first, "episode_id"),
                nws_event_id=_row_optional(first, "event_id"),
                llcrnrlat=_row_optional(first, "llcrnrlat"),
                llcrnrlon=_row_optional(first, "llcrnrlon"),
                urcrnrlat=_row_optional(first, "urcrnrlat"),
                urcrnrlon=_row_optional(first, "urcrnrlon"),
            )
        )

    summaries.sort(key=lambda e: e.time_utc, reverse=True)
    return summaries


def get_event_rows(df: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """Return all catalog rows (one per img_type) for a given event id."""
    rows = df[df["id"] == event_id]
    if rows.empty:
        raise CatalogError(f"No SEVIR event found with id={event_id!r}")
    return rows


def get_event_row_for_type(df: pd.DataFrame, event_id: str, img_type: str) -> pd.Series:
    """Return the single catalog row for (event_id, img_type)."""
    rows = get_event_rows(df, event_id)
    match = rows[rows["img_type"] == img_type]
    if match.empty:
        available = sorted(rows["img_type"].unique())
        raise CatalogError(
            f"Event {event_id!r} has no img_type={img_type!r}. "
            f"Available types: {available}"
        )
    return match.iloc[0]


def frame_offsets_minutes(row: pd.Series) -> list[int]:
    """Parse the colon-separated `minute_offsets` column into a list of ints."""
    raw = row["minute_offsets"]
    if pd.isna(raw) or raw == "":
        return []
    return [int(x) for x in str(raw).split(":")]
