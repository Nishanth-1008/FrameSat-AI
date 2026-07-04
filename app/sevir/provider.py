"""
SEVIRProvider

The data-access layer for SEVIR: given an event id and image type, loads the
requested frame(s) from the appropriate HDF5 file (resolved via the
catalog), decodes them into displayable float images, and exposes them in
the same shape the interpolation pipeline expects from ImageProvider
(uploaded files): HxWx3 uint8/float arrays.

This mirrors the `DataProvider` abstraction described in the research
report: ImageProvider handles uploads, SEVIRProvider handles the built-in
dataset. Both ultimately feed core.preprocessor / core.tensor_converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from app.config import SEVIR_DATA_DIR, SEVIR_FRAMES_PER_EVENT
from app.sevir import catalog as sevir_catalog
from app.sevir.decode import decode_frame, normalize_for_display


class SEVIRProviderError(Exception):
    """Raised for invalid event/frame requests or data access failures."""


@dataclass(frozen=True)
class FrameInfo:
    index: int
    offset_minutes: int
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class SevirFramePair:
    """Result of fetching two (or three) frames for interpolation."""

    event_id: str
    img_type: str
    frame_a_index: int
    frame_b_index: int
    image_a: np.ndarray  # HxWx3 uint8, RGB-ish, ready for the RIFE pipeline
    image_b: np.ndarray
    ground_truth_index: int | None
    ground_truth_image: np.ndarray | None


def _to_display_image(decoded_frame: np.ndarray) -> np.ndarray:
    """
    Convert a decoded single-channel SEVIR frame (H, W) into an HxWx3 uint8
    image, matching the shape ImageValidator/ImagePreprocessor expect from
    uploaded files (see core/validator.py: 3-channel requirement).
    """
    normalized = normalize_for_display(decoded_frame)
    gray_u8 = (normalized * 255.0).astype(np.uint8)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


class SEVIRProvider:
    """
    Loads SEVIR event frames for the interpolation pipeline.

    Usage:
        provider = SEVIRProvider()
        events = provider.list_events(img_types=("vil",), year=2019)
        frames = provider.list_frames(event_id, img_type="vil")
        pair = provider.get_frame_pair(event_id, "vil", frame_a=10, frame_b=12)
    """

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else SEVIR_DATA_DIR
        self._catalog_path = catalog_path
        self._catalog_df: pd.DataFrame | None = None

    @property
    def catalog(self) -> pd.DataFrame:
        if self._catalog_df is None:
            self._catalog_df = sevir_catalog.get_catalog(self._catalog_path)
        return self._catalog_df

    def reload_catalog(self) -> None:
        """Force a re-read of the catalog (e.g. after fixtures change)."""
        sevir_catalog.clear_catalog_cache()
        self._catalog_df = None

    # -- Discovery -----------------------------------------------------

    def list_events(
        self, img_types: tuple[str, ...] | None = None, year: int | None = None
    ) -> list[sevir_catalog.EventSummary]:
        return sevir_catalog.list_events(self.catalog, img_types=img_types, year=year)

    def get_event_summary(self, event_id: str) -> sevir_catalog.EventSummary:
        matches = [e for e in self.list_events() if e.event_id == event_id]
        if not matches:
            raise SEVIRProviderError(f"Unknown SEVIR event id={event_id!r}")
        return matches[0]

    def list_frames(self, event_id: str, img_type: str) -> list[FrameInfo]:
        row = sevir_catalog.get_event_row_for_type(self.catalog, event_id, img_type)
        offsets = sevir_catalog.frame_offsets_minutes(row)
        time_utc = pd.Timestamp(row["time_utc"])
        return [
            FrameInfo(
                index=i,
                offset_minutes=offset,
                timestamp=time_utc + pd.Timedelta(minutes=offset),
            )
            for i, offset in enumerate(offsets)
        ]

    # -- Frame access ----------------------------------------------------

    def _resolve_h5_path(self, file_name: str) -> Path:
        path = self.data_dir / file_name
        if not path.exists():
            raise SEVIRProviderError(
                f"SEVIR data file not found: {path}. Make sure SEVIR_DATA_DIR "
                "points at a local mirror of s3://sevir/data (see the SEVIR "
                "tutorial's download instructions), or generate fixtures for "
                "local development via app.sevir.fixtures.generate_sevir_fixture()."
            )
        return path

    def _read_raw_event_tensor(self, event_id: str, img_type: str) -> np.ndarray:
        """Read the full (L, L, T) raw integer tensor for one event/type."""
        row = sevir_catalog.get_event_row_for_type(self.catalog, event_id, img_type)
        h5_path = self._resolve_h5_path(row["file_name"])
        file_index = int(row["file_index"])

        try:
            with h5py.File(h5_path, "r") as hf:
                if img_type not in hf:
                    raise SEVIRProviderError(
                        f"HDF5 file {h5_path} has no dataset {img_type!r}"
                    )
                return hf[img_type][file_index]
        except OSError as exc:
            raise SEVIRProviderError(f"Failed to read {h5_path}: {exc}") from exc

    def get_frame_image(self, event_id: str, img_type: str, frame_index: int) -> np.ndarray:
        """Return a single decoded frame as an HxWx3 uint8 image."""
        frames = self.list_frames(event_id, img_type)
        if not (0 <= frame_index < len(frames)):
            raise SEVIRProviderError(
                f"frame_index={frame_index} out of range for event {event_id!r} "
                f"({len(frames)} frames available)"
            )

        raw_tensor = self._read_raw_event_tensor(event_id, img_type)
        raw_frame = np.asarray(raw_tensor[:, :, frame_index])
        decoded = decode_frame(raw_frame, img_type)
        return _to_display_image(decoded)

    def get_frame_pair(
        self,
        event_id: str,
        img_type: str,
        frame_a: int,
        frame_b: int,
        include_ground_truth: bool = True,
    ) -> SevirFramePair:
        """
        Fetch two frames for interpolation, plus (optionally) the true
        midpoint frame as ground truth for quality metrics, when frame_a and
        frame_b are exactly 2 apart (so a real "middle" frame exists in the
        SEVIR sequence).
        """
        frames = self.list_frames(event_id, img_type)
        n = len(frames)

        for name, idx in (("frame_a", frame_a), ("frame_b", frame_b)):
            if not (0 <= idx < n):
                raise SEVIRProviderError(
                    f"{name}={idx} out of range for event {event_id!r} ({n} frames)"
                )
        if frame_a == frame_b:
            raise SEVIRProviderError("frame_a and frame_b must differ")

        raw_tensor = self._read_raw_event_tensor(event_id, img_type)

        image_a = _to_display_image(decode_frame(np.asarray(raw_tensor[:, :, frame_a]), img_type))
        image_b = _to_display_image(decode_frame(np.asarray(raw_tensor[:, :, frame_b]), img_type))

        gt_index = None
        gt_image = None
        lo, hi = min(frame_a, frame_b), max(frame_a, frame_b)
        if include_ground_truth and hi - lo == 2:
            gt_index = lo + 1
            gt_image = _to_display_image(
                decode_frame(np.asarray(raw_tensor[:, :, gt_index]), img_type)
            )

        return SevirFramePair(
            event_id=event_id,
            img_type=img_type,
            frame_a_index=frame_a,
            frame_b_index=frame_b,
            image_a=image_a,
            image_b=image_b,
            ground_truth_index=gt_index,
            ground_truth_image=gt_image,
        )
