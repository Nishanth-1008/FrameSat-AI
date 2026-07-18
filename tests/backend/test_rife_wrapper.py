"""
Smoke test for the RIFE wrapper.

Acceptance criteria (four checks):
  1. `import app.models.rife_wrapper` succeeds.
  2. RIFEModelWrapper() instantiation succeeds (weights auto-detected).
  3. Weights load successfully (implicit in step 2 — load_model called in __init__).
  4. A dummy forward pass completes without exceptions and returns the correct shape.

Run with:
    pytest tests/backend/test_rife_wrapper.py -v
from the repository root (pytest.ini sets pythonpath = backend).
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — import succeeds
# ---------------------------------------------------------------------------
def test_import_rife_wrapper():
    """Importing the wrapper module must not raise any ImportError."""
    from app.models import rife_wrapper  # noqa: F401  # import is the test


# ---------------------------------------------------------------------------
# Acceptance criterion 2 & 3 — instantiation and weight loading succeed
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rife_model():
    """Instantiate RIFEModelWrapper once for the whole module (slow: loads weights)."""
    from app.models.rife_wrapper import RIFEModelWrapper

    return RIFEModelWrapper()  # uses the __file__-relative default weights path


def test_model_instantiation(rife_model):
    """RIFEModelWrapper() must return an object without raising."""
    from app.models.rife_wrapper import RIFEModelWrapper

    assert isinstance(rife_model, RIFEModelWrapper)


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — forward pass with dummy tensors
# ---------------------------------------------------------------------------
def test_forward_pass(rife_model):
    """
    A dummy forward pass with two [1, 3, 64, 64] tensors must complete without
    exceptions and return a tensor of shape [1, 3, 64, 64].

    64 is divisible by 32 (64/32 = 2) so no padding is required.
    """
    device = rife_model.device
    h, w = 64, 64

    frame_a = torch.rand(1, 3, h, w, device=device)
    frame_b = torch.rand(1, 3, h, w, device=device)

    result = rife_model.interpolate(frame_a, frame_b)

    assert result is not None, "interpolate() returned None"
    assert result.shape == torch.Size([1, 3, h, w]), (
        f"Expected output shape [1, 3, {h}, {w}], got {result.shape}"
    )
