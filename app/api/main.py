# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

from app.api.routes.system import router as system_router
from app.api.routes.interpolate import router as interpolate_router

from app.config import OUTPUT_DIR

app = FastAPI(
    title="FrameSat AI API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)

app.include_router(system_router)
app.include_router(interpolate_router)


@app.get("/")
def root():
    return {
        "message": "FrameSat AI Backend Running"
    }