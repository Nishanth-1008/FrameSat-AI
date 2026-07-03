import json
from datetime import datetime
from pathlib import Path
from typing import Any, Union
from core.constants import SUPPORTED_FORMATS

def ensure_directory(path: Union[str, Path]) -> None:
    """Ensure that a directory exists, creating parent directories if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size(path: Union[str, Path]) -> int:
    """Return the size of a file in bytes."""
    return Path(path).stat().st_size


def is_supported_format(filename: Union[str, Path]) -> bool:
    """Check if the filename has a supported image format extension."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_FORMATS


def generate_timestamp() -> str:
    """Generate a timestamp string suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data to a JSON file, ensuring the parent directory exists."""
    path_obj = Path(path)
    ensure_directory(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
