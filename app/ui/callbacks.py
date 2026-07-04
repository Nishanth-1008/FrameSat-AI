from pathlib import Path
import sys
import tempfile
import time

# ------------------------------------------------------------
# Project Root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from app.services.interpolation_service import InterpolationService

service = InterpolationService()


def generate(frame_a, frame_b):

    if frame_a is None or frame_b is None:
        return (
            None,
            frame_a,
            frame_b,
            "Please upload both frames.",
            "--",
            "--",
            None,
        )

    try:

        start = time.perf_counter()

        output_path = Path(tempfile.gettempdir()) / "framesat_result.png"

        service.interpolate(
            frame_a,
            frame_b,
            str(output_path),
        )

        runtime = time.perf_counter() - start

        return (
            str(output_path),
            frame_a,
            frame_b,
            "Completed",
            f"{runtime:.2f} sec",
            "512 × 512",
            str(output_path),
        )

    except Exception as e:

        return (
            None,
            frame_a,
            frame_b,
            str(e),
            "--",
            "--",
            None,
        )