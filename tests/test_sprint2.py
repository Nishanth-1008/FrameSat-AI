import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEVICE, LOG_LEVEL, BASE_DIR
from app.logger import logger
from core.constants import SUPPORTED_FORMATS, DEFAULT_INTERPOLATION_TIME, MODEL_NAME, APP_NAME
from core.exceptions import ValidationError, ModelLoadError, InterpolationError
from core.utils import (
    ensure_directory,
    get_file_size,
    is_supported_format,
    generate_timestamp,
    save_json,
    load_json,
)

def run_tests():
    print("--- 1. Testing Config & Logger ---")
    logger.info(f"Running on device: {DEVICE}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    
    print("\n--- 2. Testing Constants ---")
    print(f"Supported Formats: {SUPPORTED_FORMATS}")
    print(f"Default Interpolation Time: {DEFAULT_INTERPOLATION_TIME}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"App Name: {APP_NAME}")
    
    print("\n--- 3. Testing Exceptions ---")
    try:
        raise ValidationError("Test validation error")
    except ValidationError as e:
        print(f"ValidationError caught: {e}")
        
    try:
        raise ModelLoadError("Test model load error")
    except ModelLoadError as e:
        print(f"ModelLoadError caught: {e}")
        
    try:
        raise InterpolationError("Test interpolation error")
    except InterpolationError as e:
        print(f"InterpolationError caught: {e}")

    print("\n--- 4. Testing Utilities ---")
    temp_dir = BASE_DIR / "logs" / "temp_test_dir"
    ensure_directory(temp_dir)
    print(f"Directory {temp_dir} exists: {temp_dir.exists()}")
    
    test_json_file = temp_dir / "test.json"
    test_data = {"test_key": "test_value", "timestamp": generate_timestamp()}
    save_json(test_data, test_json_file)
    print(f"JSON saved to {test_json_file}")
    
    loaded_data = load_json(test_json_file)
    print(f"JSON loaded: {loaded_data}")
    assert loaded_data["test_key"] == "test_value"
    
    size = get_file_size(test_json_file)
    print(f"File size: {size} bytes")
    assert size > 0
    
    # Test extension support checks
    print(f"is 'test.png' supported? {is_supported_format('test.png')}")
    print(f"is 'test.txt' supported? {is_supported_format('test.txt')}")
    
    # Cleanup test files
    test_json_file.unlink()
    temp_dir.rmdir()
    print("Cleanup completed.")
    print("\n✅ All tests passed successfully!")

if __name__ == "__main__":
    run_tests()
