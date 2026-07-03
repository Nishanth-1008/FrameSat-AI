from pathlib import Path
import sys

# pyrefly: ignore [missing-import]
import gradio as gr

# ------------------------------------------------------------------
# Project Root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from app.ui.sidebar import create_sidebar
# pyrefly: ignore [missing-import]
from app.ui.callbacks import generate


def create_dashboard():

    with gr.Blocks(
        title="FrameSat AI",
        css="app/ui/styles.css",
        fill_height=True,
    ) as demo:

        with gr.Row(equal_height=True):

            # ==========================================================
            # Sidebar
            # ==========================================================

            with gr.Column(scale=1, min_width=300):

                create_sidebar()

            # ==========================================================
            # Main Content
            # ==========================================================

            with gr.Column(scale=4):

                # ---------------- Header ----------------

                gr.HTML("""
                <div class="header">

                    <div>

                        <h1>Temporal Interpolation Engine</h1>

                        <p>
                            AI-powered satellite frame interpolation using Practical-RIFE
                        </p>

                    </div>

                    <div class="header-badge">

                        Practical-RIFE · PyTorch

                    </div>

                </div>
                """)

                # ---------------- Upload Section ----------------

                with gr.Group():

                    gr.Markdown("## Upload Frames")

                    with gr.Row():

                        frame_a = gr.Image(
                            type="filepath",
                            label="Frame A",
                        )

                        frame_b = gr.Image(
                            type="filepath",
                            label="Frame B",
                        )

                    generate_btn = gr.Button(
                        "🚀 Generate Intermediate Frame",
                        variant="primary",
                        size="lg",
                    )

                # ---------------- Results ----------------

                with gr.Group():

                    gr.Markdown("## Interpolation Results")

                    with gr.Row():

                        preview_a = gr.Image(
                            label="Frame A",
                            interactive=False,
                        )

                        generated = gr.Image(
                            label="Generated Frame",
                            interactive=False,
                        )

                        preview_b = gr.Image(
                            label="Frame B",
                            interactive=False,
                        )

                # ---------------- Statistics ----------------

                with gr.Group():

                    gr.Markdown("## Runtime Information")

                    with gr.Row():

                        status = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False,
                        )

                        runtime = gr.Textbox(
                            label="Inference Time",
                            value="--",
                            interactive=False,
                        )

                        resolution = gr.Textbox(
                            label="Resolution",
                            value="--",
                            interactive=False,
                        )

                download = gr.File(
                    label="Download Output",
                )

                # ---------------- Callback ----------------

                generate_btn.click(
                    fn=generate,
                    inputs=[
                        frame_a,
                        frame_b,
                    ],
                    outputs=[
                        generated,
                        preview_a,
                        preview_b,
                        status,
                        runtime,
                        resolution,
                        download,
                    ],
                )

    return demo