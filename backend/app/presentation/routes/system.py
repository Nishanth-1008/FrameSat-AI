# pyrefly: ignore [missing-import]
from fastapi import APIRouter

from app.container import container

router = APIRouter(prefix="/system", tags=["System"])


@router.get("")
def system():
    settings = container.settings

    return {
        "application": settings.APP_NAME,
        "version": settings.VERSION,
        "device": settings.DEVICE,
        "model": settings.MODEL_NAME,
    }