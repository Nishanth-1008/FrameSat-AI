"""
Integration tests for the SEVIR API routes.

We build a minimal FastAPI app that mounts the SEVIR router directly,
rather than importing backend_api.api, because api.py's module-level
`InterpolationService()` (via RIFEWrapper) requires the external
Practical-RIFE repo + pretrained weights that only exist on the user's
own machine, not in this sandbox. The routes under test here are pure
SEVIR plumbing and don't depend on the real model.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend_api.sevir_routes as sevir_routes
from app.sevir.provider import SEVIRProvider


class _MockInterpolationService:
    """Writes frame_a as the 'interpolated' output, avoiding real RIFE/GPU."""

    def interpolate(self, frame1_path, frame2_path, output_path):
        shutil.copyfile(frame1_path, output_path)
        return output_path


@pytest.fixture()
def sevir_api_client(sevir_fixture_root, monkeypatch, tmp_path):
    root, catalog_path = sevir_fixture_root

    # Point the module-level provider at our fixture data.
    test_provider = SEVIRProvider(catalog_path=catalog_path, data_dir=root / "data")
    monkeypatch.setattr(sevir_routes, "_provider", test_provider)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(sevir_routes, "OUTPUT_DIR", output_dir)

    app = FastAPI()
    app.include_router(sevir_routes.router)

    mock_service = _MockInterpolationService()
    sevir_routes.register_interpolate_sevir_route(
        app,
        mock_service,
        is_busy_getter=lambda: False,
        is_busy_setter=lambda v: None,
        status_getter=lambda: "READY",
    )

    return TestClient(app)


def _first_event_id(client) -> str:
    resp = client.get("/datasets/sevir/events")
    assert resp.status_code == 200
    return resp.json()["events"][0]["event_id"]


def test_list_datasets(sevir_api_client):
    resp = sevir_api_client.get("/datasets")
    assert resp.status_code == 200
    assert resp.json() == ["sevir"]


def test_list_events_returns_all_fixture_events(sevir_api_client):
    resp = sevir_api_client.get("/datasets/sevir/events")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert len(body["events"]) == 5


def test_list_events_pagination(sevir_api_client):
    resp = sevir_api_client.get("/datasets/sevir/events", params={"page": 1, "per_page": 2})
    body = resp.json()
    assert len(body["events"]) == 2
    assert body["page"] == 1
    assert body["per_page"] == 2


def test_list_events_filters_by_year(sevir_api_client):
    resp = sevir_api_client.get("/datasets/sevir/events", params={"year": 2099})
    assert resp.json()["total"] == 0


def test_get_event_frames(sevir_api_client):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.get(f"/datasets/sevir/events/{event_id}/frames", params={"img_type": "vil"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_id"] == event_id
    assert len(body["frames"]) == 49


def test_get_event_frames_unknown_event_returns_404(sevir_api_client):
    resp = sevir_api_client.get("/datasets/sevir/events/does-not-exist/frames")
    assert resp.status_code == 404


def test_get_event_frames_invalid_img_type_returns_400(sevir_api_client):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.get(
        f"/datasets/sevir/events/{event_id}/frames", params={"img_type": "not_real"}
    )
    assert resp.status_code == 400


def test_frame_preview_returns_png_url(sevir_api_client):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.get(
        f"/datasets/sevir/events/{event_id}/frames/5/preview", params={"img_type": "vil"}
    )
    assert resp.status_code == 200
    assert resp.json()["preview_url"].endswith(".png")


def test_interpolate_sevir_success_with_ground_truth_metrics(sevir_api_client):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.post(
        "/interpolate/sevir",
        json={"event_id": event_id, "img_type": "vil", "frame_a": 4, "frame_b": 6},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_id"] == event_id
    assert body["ground_truth_frame"] == 5
    assert body["psnr"] is not None
    assert body["ssim"] is not None
    assert "image_url" in body


def test_interpolate_sevir_no_ground_truth_when_frames_not_adjacent(sevir_api_client):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.post(
        "/interpolate/sevir",
        json={"event_id": event_id, "img_type": "vil", "frame_a": 0, "frame_b": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ground_truth_frame"] is None
    assert body["psnr"] is None
    assert body["ssim"] is None


def test_interpolate_sevir_invalid_event_returns_404(sevir_api_client):
    resp = sevir_api_client.post(
        "/interpolate/sevir",
        json={"event_id": "does-not-exist", "img_type": "vil", "frame_a": 0, "frame_b": 2},
    )
    assert resp.status_code == 404


def test_interpolate_sevir_invalid_img_type_returns_400(sevir_api_client):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.post(
        "/interpolate/sevir",
        json={"event_id": event_id, "img_type": "lght", "frame_a": 0, "frame_b": 2},
    )
    assert resp.status_code == 400


def test_interpolate_sevir_returns_503_when_busy(sevir_fixture_root, monkeypatch, tmp_path):
    root, catalog_path = sevir_fixture_root
    test_provider = SEVIRProvider(catalog_path=catalog_path, data_dir=root / "data")
    monkeypatch.setattr(sevir_routes, "_provider", test_provider)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(sevir_routes, "OUTPUT_DIR", output_dir)

    app = FastAPI()
    app.include_router(sevir_routes.router)
    sevir_routes.register_interpolate_sevir_route(
        app,
        _MockInterpolationService(),
        is_busy_getter=lambda: True,  # simulate an in-flight request
        is_busy_setter=lambda v: None,
        status_getter=lambda: "READY",
    )
    client = TestClient(app)
    event_id = _first_event_id(client)

    resp = client.post(
        "/interpolate/sevir",
        json={"event_id": event_id, "img_type": "vil", "frame_a": 0, "frame_b": 2},
    )
    assert resp.status_code == 503


def test_interpolate_sevir_returns_500_when_service_in_error_state(
    sevir_fixture_root, monkeypatch, tmp_path
):
    root, catalog_path = sevir_fixture_root
    test_provider = SEVIRProvider(catalog_path=catalog_path, data_dir=root / "data")
    monkeypatch.setattr(sevir_routes, "_provider", test_provider)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(sevir_routes, "OUTPUT_DIR", output_dir)

    app = FastAPI()
    app.include_router(sevir_routes.router)
    sevir_routes.register_interpolate_sevir_route(
        app,
        None,  # service failed to initialize
        is_busy_getter=lambda: False,
        is_busy_setter=lambda v: None,
        status_getter=lambda: "ERROR",
    )
    client = TestClient(app)
    event_id = _first_event_id(client)

    resp = client.post(
        "/interpolate/sevir",
        json={"event_id": event_id, "img_type": "vil", "frame_a": 0, "frame_b": 2},
    )
    assert resp.status_code == 500


def test_interpolate_sevir_cleans_up_temp_files(sevir_api_client, monkeypatch):
    event_id = _first_event_id(sevir_api_client)
    resp = sevir_api_client.post(
        "/interpolate/sevir",
        json={"event_id": event_id, "img_type": "vil", "frame_a": 4, "frame_b": 6},
    )
    assert resp.status_code == 200
    temp_dir = sevir_routes.OUTPUT_DIR / "sevir_temp"
    # temp_a/temp_b should have been removed in the `finally` block
    assert list(temp_dir.glob("*_a.png")) == []
    assert list(temp_dir.glob("*_b.png")) == []
