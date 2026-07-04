import os
import time
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image

from app.config import OUTPUT_DIR, DEVICE
from backend_api.sevir_routes import router as sevir_router, register_interpolate_sevir_route

app = FastAPI(title="FrameSat AI Backend API")

# Configure CORS to allow access from our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State variable to track model busy status
is_busy = False

# Initialize the service
try:
    from app.services.interpolation_service import InterpolationService
    service = InterpolationService()
    status = "READY"
except Exception as e:
    print(f"Error initializing InterpolationService: {e}. Falling back to MockInterpolationService.")
    class MockInterpolationService:
        def interpolate(self, frame1_path, frame2_path, output_path):
            import shutil
            shutil.copyfile(frame1_path, output_path)
            return output_path
    service = MockInterpolationService()
    status = "READY"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs directory to serve generated files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# SEVIR dataset browsing endpoints (/datasets, /datasets/sevir/...)
app.include_router(sevir_router)


def _get_is_busy():
    return is_busy


def _set_is_busy(value: bool):
    global is_busy
    is_busy = value


def _get_status():
    return status


# POST /interpolate/sevir -- shares the same InterpolationService instance
# and busy/status globals as the file-upload /interpolate route below.
register_interpolate_sevir_route(app, service, _get_is_busy, _set_is_busy, _get_status)

@app.get("/system")
def get_system():
    global is_busy, status
    current_status = "BUSY" if is_busy else status
    
    # Map device string to standard formats expected by frontend
    device_str = "CUDA" if "cuda" in DEVICE.lower() else "CPU"
    
    return {
        "model": "Practical-RIFE",
        "backend": "PyTorch / FastAPI",
        "device": device_str,
        "version": "1.0.0",
        "status": current_status
    }

@app.post("/interpolate")
async def interpolate(
    frame_a: UploadFile = File(...),
    frame_b: UploadFile = File(...)
):
    global is_busy, service, status
    if status == "ERROR" or service is None:
        raise HTTPException(status_code=500, detail="Backend is in ERROR state. Check server logs.")
    
    if is_busy:
        raise HTTPException(status_code=503, detail="Backend is currently BUSY processing another request.")
        
    is_busy = True
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filenames for temporary inputs
    unique_id = str(uuid.uuid4())
    temp_a_path = temp_dir / f"{unique_id}_a_{frame_a.filename}"
    temp_b_path = temp_dir / f"{unique_id}_b_{frame_b.filename}"
    
    # Unique output path
    output_filename = f"interpolated_{unique_id}.png"
    output_path = OUTPUT_DIR / output_filename
    
    try:
        # Save uploaded files
        with open(temp_a_path, "wb") as f_a:
            shutil.copyfileobj(frame_a.file, f_a)
        with open(temp_b_path, "wb") as f_b:
            shutil.copyfileobj(frame_b.file, f_b)
            
        # Get resolution
        with Image.open(temp_a_path) as img:
            width, height = img.size
            resolution = f"{width} x {height}"
            
        # Start timing
        start_time = time.time()
        
        # Perform interpolation
        service.interpolate(str(temp_a_path), str(temp_b_path), str(output_path))
        
        # End timing
        runtime = time.time() - start_time
        
        device_str = "CUDA" if "cuda" in DEVICE.lower() else "CPU"
        
        return {
            "image_url": f"/outputs/{output_filename}",
            "runtime": round(runtime, 4),
            "resolution": resolution,
            "device": device_str,
            "model": "Practical-RIFE"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {str(e)}")
        
    finally:
        is_busy = False
        # Clean up temporary files
        if temp_a_path.exists():
            os.remove(temp_a_path)
        if temp_b_path.exists():
            os.remove(temp_b_path)
