from pathlib import Path
from PIL import Image

from core.data_loader import load_pair
from core.validator import validate_pair

# Ensure sample inputs exist
input_dir = Path("assets/sample_inputs")
input_dir.mkdir(parents=True, exist_ok=True)

frame_a_path = input_dir / "frame_a.png"
frame_b_path = input_dir / "frame_b.png"

if not frame_a_path.exists():
    img_a = Image.new("RGB", (512, 512), color="blue")
    img_a.save(frame_a_path)
    print(f"Created sample input: {frame_a_path}")

if not frame_b_path.exists():
    img_b = Image.new("RGB", (512, 512), color="green")
    img_b.save(frame_b_path)
    print(f"Created sample input: {frame_b_path}")

# Run validation
validate_pair(
    "assets/sample_inputs/frame_a.png",
    "assets/sample_inputs/frame_b.png",
)

a, b = load_pair(
    "assets/sample_inputs/frame_a.png",
    "assets/sample_inputs/frame_b.png",
)

print(a.shape)
print(b.shape)
