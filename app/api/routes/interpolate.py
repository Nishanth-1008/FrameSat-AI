import shutil
import tempfile
import time
import uuid
from pathlib import Path

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, File, HTTPException, UploadFile
# pyrefly: ignore [missing-import]
from PIL import Image

from app.api.schemas.interpolate import InterpolationResponse
from app.config import DEVICE, OUTPUT_DIR
from app.services.interpolation_service import InterpolationService

router = APIRouter(tags=["Interpolation"])

service = InterpolationService()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/interpolate",
    response_model=InterpolationResponse,
)
async def interpolate(
    frame_a: UploadFile = File(...),
    frame_b: UploadFile = File(...),
):
    temp_dir = Path(tempfile.gettempdir()) / "framesat_ai"
    temp_dir.mkdir(exist_ok=True)

    uid = uuid.uuid4().hex

    input_a = temp_dir / f"{uid}_a.png"
    input_b = temp_dir / f"{uid}_b.png"

    output = OUTPUT_DIR / f"{uid}.png"

    try:

        with open(input_a, "wb") as f:
            shutil.copyfileobj(frame_a.file, f)

        with open(input_b, "wb") as f:
            shutil.copyfileobj(frame_b.file, f)

        with Image.open(input_a) as img:
            w, h = img.size

        start = time.perf_counter()

        service.interpolate(
            str(input_a),
            str(input_b),
            str(output),
        )

        runtime = time.perf_counter() - start

        return InterpolationResponse(
            image_url=f"/outputs/{output.name}",
            runtime=round(runtime, 3),
            resolution=f"{w} × {h}",
            device="CUDA" if "cuda" in DEVICE.lower() else "CPU",
            model="Practical-RIFE",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        input_a.unlink(missing_ok=True)
        input_b.unlink(missing_ok=True)