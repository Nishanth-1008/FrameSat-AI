# pyrefly: ignore [missing-import]
from fastapi import APIRouter

from app.presentation.routes.health import router as health_router
from app.presentation.routes.system import router as system_router
from app.presentation.routes.root import router as root_router

api_router = APIRouter()

api_router.include_router(root_router)

api_router.include_router(
    health_router,
    prefix="/api/v1",
)

api_router.include_router(
    system_router,
    prefix="/api/v1",
)
