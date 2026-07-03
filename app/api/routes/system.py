# pyrefly: ignore [missing-import]
from fastapi import APIRouter

from app.config import DEVICE

router = APIRouter()


@router.get("/system")
def get_system():
    device = "CUDA" if "cuda" in DEVICE.lower() else "CPU"

    return {
        "model": "Practical-RIFE",
        "backend": "PyTorch / FastAPI",
        "device": device,
        "version": "1.0.0",
        "status": "READY",
    }