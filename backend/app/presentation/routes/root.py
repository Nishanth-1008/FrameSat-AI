# pyrefly: ignore [missing-import]
from fastapi import APIRouter

router = APIRouter(tags=["Root"])


@router.get("/")
def root():
    return {
        "application": "FrameSat AI",
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1",
    }
