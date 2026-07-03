from pathlib import Path
import sys

# pyrefly: ignore [missing-import]
import gradio as gr
# pyrefly: ignore [missing-import]
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_sidebar():

    device = "CUDA" if torch.cuda.is_available() else "CPU"

    html = f"""
<div class="sidebar">

    <h1>🛰 FrameSat AI</h1>

    <div class="subtitle">
        Satellite Frame Interpolation Platform
    </div>

    <div class="status-chip">
        <div class="status-dot"></div>
        AI CORE ACTIVE
    </div>

    <div class="info-card">
        <span class="label">MODEL</span>
        <span class="value">Practical-RIFE</span>
    </div>

    <div class="info-card">
        <span class="label">BACKEND</span>
        <span class="value">PyTorch</span>
    </div>

    <div class="info-card">
        <span class="label">DEVICE</span>
        <span class="value">{device}</span>
    </div>

    <div class="info-card">
        <span class="label">VERSION</span>
        <span class="value">v1.0.0</span>
    </div>

    <div class="info-card">
        <span class="label">STATUS</span>
        <span class="value">READY</span>
    </div>

</div>
"""

    return gr.HTML(html)