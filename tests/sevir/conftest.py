"""
Shared pytest fixtures for SEVIR tests.

We never touch AWS S3 or real SEVIR HDF5 files here (no network access in
this sandbox / CI). Instead we generate a small synthetic dataset that
matches the real SEVIR CATALOG.csv schema and HDF5 layout, exercised via
app.sevir.fixtures.generate_sevir_fixture(). Real end-to-end validation
against actual SEVIR data happens separately, on a machine with S3 access.
"""

from __future__ import annotations

import pytest

from app.sevir.catalog import clear_catalog_cache
from app.sevir.fixtures import generate_sevir_fixture
from app.sevir.provider import SEVIRProvider


@pytest.fixture()
def sevir_fixture_root(tmp_path):
    root = tmp_path / "sevir_fixture"
    catalog_path = generate_sevir_fixture(root, n_events=5, year=2019)
    yield root, catalog_path
    clear_catalog_cache()


@pytest.fixture()
def sevir_provider(sevir_fixture_root) -> SEVIRProvider:
    root, catalog_path = sevir_fixture_root
    return SEVIRProvider(catalog_path=catalog_path, data_dir=root / "data")
