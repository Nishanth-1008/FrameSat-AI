from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from app.ui.dashboard import create_dashboard

demo = create_dashboard()

if __name__ == "__main__":
    demo.launch()