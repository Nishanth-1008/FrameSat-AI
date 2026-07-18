# pyrefly: ignore [missing-import]
from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
import os

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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Mount outputs directory as static path
outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "outputs")
os.makedirs(outputs_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=outputs_dir), name="static")

register_exception_handlers(app)
