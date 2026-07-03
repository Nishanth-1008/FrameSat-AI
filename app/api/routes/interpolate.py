import os
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

router = APIRouter()

service = InterpolationService()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)