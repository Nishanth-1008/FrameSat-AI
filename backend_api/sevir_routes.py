"""
SEVIR API Routes

Endpoints for browsing the SEVIR dataset (events, frames) and running
interpolation against SEVIR-provided frame pairs, as distinct from the
existing file-upload `/interpolate` endpoint in api.py.

Routes:
    GET  /datasets                                  -> list available datasets
    GET  /datasets/sevir/events                      -> list/browse events
    GET  /datasets/sevir/events/{event_id}/frames     -> frame timeline for an event
    POST /interpolate/sevir                           -> interpolate two SEVIR frames
"""

from __future__ import annotations

import time
import uuid

import cv2
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import (
    DEVICE,
    OUTPUT_DIR,
    SEVIR_DEFAULT_IMG_TYPE,
    SEVIR_EVENTS_DEFAULT_PAGE_SIZE,
    SEVIR_EVENTS_MAX_PAGE_SIZE,
    SEVIR_RASTER_TYPES,
)
from app.sevir.catalog import CatalogError
from app.sevir.metrics import compute_all_metrics
from app.sevir.provider import SEVIRProvider, SEVIRProviderError

router = APIRouter()

_provider = SEVIRProvider()


class SevirInterpolateRequest(BaseModel):
    event_id: str
    img_type: str = Field(default=SEVIR_DEFAULT_IMG_TYPE)
    frame_a: int
    frame_b: int


def _provider_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, (SEVIRProviderError, CatalogError)):
        # Distinguish "not found" from other bad input for a nicer status code.
        message = str(exc)
        if "not found" in message or "Unknown SEVIR event" in message or "No SEVIR event" in message:
            return HTTPException(status_code=404, detail=message)
        return HTTPException(status_code=400, detail=message)
    return HTTPException(status_code=500, detail=f"Unexpected SEVIR error: {exc}")


@router.get("/datasets")
def list_datasets():
    return ["sevir"]


@router.get("/datasets/sevir/events")
def list_sevir_events(
    year: int | None = Query(default=None),
    img_type: str | None = Query(default=None, description="Require this img_type to be available"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=SEVIR_EVENTS_DEFAULT_PAGE_SIZE, ge=1, le=SEVIR_EVENTS_MAX_PAGE_SIZE),
):
    try:
        img_types = (img_type,) if img_type else None
        events = _provider.list_events(img_types=img_types, year=year)
    except (SEVIRProviderError, CatalogError) as exc:
        raise _provider_error_to_http(exc)

    total = len(events)
    start = (page - 1) * per_page
    end = start + per_page
    page_events = events[start:end]

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "events": [e.to_dict() for e in page_events],
    }


@router.get("/datasets/sevir/events/{event_id}/frames")
def get_sevir_event_frames(
    event_id: str,
    img_type: str = Query(default=SEVIR_DEFAULT_IMG_TYPE),
):
    if img_type not in SEVIR_RASTER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"img_type must be one of {SEVIR_RASTER_TYPES} (got {img_type!r})",
        )
    try:
        frames = _provider.list_frames(event_id, img_type)
    except (SEVIRProviderError, CatalogError) as exc:
        raise _provider_error_to_http(exc)

    return {
        "event_id": event_id,
        "img_type": img_type,
        "frames": [
            {
                "index": f.index,
                "offset_minutes": f.offset_minutes,
                "timestamp": f.timestamp.isoformat(),
            }
            for f in frames
        ],
    }


@router.get("/datasets/sevir/events/{event_id}/frames/{frame_index}/preview")
def get_sevir_frame_preview(
    event_id: str,
    frame_index: int,
    img_type: str = Query(default=SEVIR_DEFAULT_IMG_TYPE),
):
    """Return a single decoded SEVIR frame as a PNG, for timeline thumbnails."""
    try:
        image = _provider.get_frame_image(event_id, img_type, frame_index)
    except (SEVIRProviderError, CatalogError) as exc:
        raise _provider_error_to_http(exc)

    preview_dir = OUTPUT_DIR / "sevir_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{event_id}_{img_type}_{frame_index}.png"
    path = preview_dir / filename
    if not path.exists():
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return {"preview_url": f"/outputs/sevir_previews/{filename}"}


def register_interpolate_sevir_route(app, service, is_busy_getter, is_busy_setter, status_getter):
    """
    Registers POST /interpolate/sevir on `app`, sharing the same
    InterpolationService/busy-state as the file-upload /interpolate route in
    api.py (single global model instance, single in-flight inference at a
    time -- see research report's Concurrency notes).
    """

    @app.post("/interpolate/sevir")
    def interpolate_sevir(payload: SevirInterpolateRequest):
        if status_getter() == "ERROR" or service is None:
            raise HTTPException(status_code=500, detail="Backend is in ERROR state. Check server logs.")
        if is_busy_getter():
            raise HTTPException(status_code=503, detail="Backend is currently BUSY processing another request.")

        if payload.img_type not in SEVIR_RASTER_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"img_type must be one of {SEVIR_RASTER_TYPES} (got {payload.img_type!r})",
            )

        is_busy_setter(True)
        try:
            pair = _provider.get_frame_pair(
                payload.event_id, payload.img_type, payload.frame_a, payload.frame_b
            )
        except (SEVIRProviderError, CatalogError) as exc:
            is_busy_setter(False)
            raise _provider_error_to_http(exc)

        temp_dir = OUTPUT_DIR / "sevir_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        unique_id = str(uuid.uuid4())
        temp_a_path = temp_dir / f"{unique_id}_a.png"
        temp_b_path = temp_dir / f"{unique_id}_b.png"
        output_filename = f"interpolated_sevir_{unique_id}.png"
        output_path = OUTPUT_DIR / output_filename

        try:
            cv2.imwrite(str(temp_a_path), cv2.cvtColor(pair.image_a, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(temp_b_path), cv2.cvtColor(pair.image_b, cv2.COLOR_RGB2BGR))

            height, width = pair.image_a.shape[:2]
            resolution = f"{width} x {height}"

            start_time = time.time()
            service.interpolate(str(temp_a_path), str(temp_b_path), str(output_path))
            runtime = time.time() - start_time

            metrics = compute_all_metrics(
                cv2.cvtColor(cv2.imread(str(output_path)), cv2.COLOR_BGR2RGB),
                pair.ground_truth_image,
            )

            device_str = "CUDA" if "cuda" in DEVICE.lower() else "CPU"

            return {
                "image_url": f"/outputs/{output_filename}",
                "runtime": round(runtime, 4),
                "resolution": resolution,
                "device": device_str,
                "model": "Practical-RIFE",
                "event_id": pair.event_id,
                "img_type": pair.img_type,
                "frame_a": pair.frame_a_index,
                "frame_b": pair.frame_b_index,
                "ground_truth_frame": pair.ground_truth_index,
                **metrics.to_dict(),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SEVIR interpolation failed: {exc}")
        finally:
            is_busy_setter(False)
            for p in (temp_a_path, temp_b_path):
                if p.exists():
                    p.unlink()
