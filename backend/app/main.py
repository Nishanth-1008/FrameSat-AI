# pyrefly: ignore [missing-import]
from fastapi import FastAPI

from app.lifespan import lifespan
from app.presentation.api import api_router
from app.presentation.handlers.exceptions import register_exception_handlers
from app.shared.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

app.include_router(api_router)

register_exception_handlers(app)