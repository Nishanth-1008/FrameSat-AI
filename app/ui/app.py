import sys
from pathlib import Path
import tempfile
# pyrefly: ignore [missing-import]
import gradio as gr

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

# pyrefly: ignore [missing-import]
from app.services.interpolation_service import InterpolationService

service = InterpolationService()


def interpolate(frame1, frame2):
    if frame1 is None or frame2 is None:
        return None

    output_path = Path(tempfile.gettempdir()) / "framesat_result.png"

    service.interpolate(
        frame1,
        frame2,
        str(output_path),
    )

    return str(output_path)


demo = gr.Interface(
    fn=interpolate,
    inputs=[
        gr.Image(type="filepath", label="Frame A"),
        gr.Image(type="filepath", label="Frame B"),
    ],
    outputs=gr.Image(type="filepath", label="Interpolated Frame"),
    title="FrameSat-AI",
    description="AI-powered satellite frame interpolation using Practical-RIFE",
)

if __name__ == "__main__":
    demo.launch()