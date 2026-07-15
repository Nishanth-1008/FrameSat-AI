from __future__ import annotations

from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI

from app.container import container


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.
    """

    container.logger.info("Starting FrameSat AI")

    yield

    container.logger.info("Stopping FrameSat AI")
