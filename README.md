# FrameSat AI

> **Satellite Temporal Intelligence Platform** — Reconstructing missing environmental observations using advanced deep learning.

---

## Project Overview

**FrameSat AI** is an AI-powered geospatial temporal intelligence platform that reconstructs missing observations between satellite acquisitions. By generating high-fidelity intermediate frames, FrameSat AI increases the apparent temporal resolution of Earth Observation (EO) datasets, empowering disaster analysts, remote sensing researchers, and meteorological agencies to monitor rapidly evolving events.

### The Problem

Satellite imagery is inherently discrete. A given satellite may capture an area at 10:00, 10:10, and 10:20, leaving 10-minute observation gaps. Critical environmental events — such as cyclones, thunderstorms, floods, wildfires, and cloud formations — evolve continuously during these gaps. The lack of visual continuity reduces situational awareness and limits post-event analysis.

FrameSat AI bridges this gap by leveraging frame interpolation deep learning models (specifically RIFE) adapted for high-bitrate scientific remote sensing datasets, reconstructing plausible intermediate states to create a continuous observation pipeline.

---

## Tech Stack

FrameSat AI is designed with clean architecture principles and a modular backend:

*   **Core & Deep Learning**: Python 3.10+, PyTorch, TorchVision
*   **Geospatial & Image Processing**: `rasterio` (GDAL-based GeoTIFF handling), `xarray`, `scikit-image`, `h5py`, `tifffile`, `numpy`, OpenCV, Pillow
*   **Web Framework & API**: FastAPI, Uvicorn, Pydantic, `pydantic-settings`
*   **User Interface**: Gradio (for direct modeling demo), Frontend Web Application (planned)
*   **Utilities & Tooling**: `python-dotenv`, `tqdm`, `pytest` (testing), `ruff` (linter), `black` (formatter)

---

## Folder Structure

The workspace follows the clean directory hierarchy outlined in the system architecture (ADR-001):

```text
FrameSat-AI/
├── assets/                # Static assets, input/output samples
│   ├── sample_inputs/     # Directory for test satellite input data
│   └── sample_outputs/    # Directory for generated interpolation results
├── backend/               # Python application package source
│   ├── app/               # Main application container
│   │   ├── domain/        # Enterprise domain rules (entities, value objects, interfaces)
│   │   ├── application/   # Orchestration & use cases (services, DTOs)
│   │   ├── infrastructure/# Technical details (logging engine, file storage, AI model loader)
│   │   │   └── logging/   # Infra-layer logging wrapper
│   │   ├── presentation/  # Interface layer (FastAPI endpoints, routers, HTTP error handlers)
│   │   ├── shared/        # Cross-cutting concerns (shared configuration mapping)
│   │   ├── config.py      # Typed environment configuration
│   │   ├── logger.py      # Reusable logging core
│   │   ├── container.py   # DI composition root
│   │   └── main.py        # API Entrypoint
│   ├── tests/             # Backend test suites
│   └── .env               # Backend environment file
├── datasets/              # Geospatial and sample datasets
├── docs/                  # System documentation & Architecture Decision Records (ADRs)
│   ├── architecture/      # Architectural definitions and ADRs
│   ├── development/       # Standards, workflows, definition of done
│   └── roadmap/           # Sprint planning and milestones
├── frontend/              # Next.js/Vite Web application (planned)
├── weights/               # Local cache for AI model checkpoints (e.g. RIFE model weights)
├── requirements.txt       # Combined workspace python dependencies
├── .gitignore             # Git ignore file configurations
└── .env.example           # Workspace configuration template
```

---

## Getting Started

### Prerequisites

*   Python 3.10 or higher
*   C compiler & GDAL developer libraries (required for compiling `rasterio`)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Nishanth-1008/FrameSat-AI.git
    cd FrameSat-AI
    ```

2.  **Set Up Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Copy `.env.example` to create your local `.env` configuration file:
    ```bash
    copy .env.example .env
    ```

---

## Run & Usage Instructions

### Starting the Backend API

To run the FastAPI server with auto-reload:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the documentation at:
*   Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
*   Alternative API Docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Running Tests

Run the test suite using `pytest` from the root directory:

```bash
pytest
```

---

## Platform Visuals

### System User Interface
![FrameSat AI User Interface Mockup](https://raw.githubusercontent.com/Nishanth-1008/FrameSat-AI/main/docs/ui/mockup_placeholder.png)
*Placeholder: A high-fidelity web dashboard displaying temporal interpolation split-screens, datasets explorer, and export options.*

---

## Project Roadmap

*   [x] **Sprint 1: Project Foundation**
    *   Setup workspace directories, dependencies, and git constraints.
    *   Implement typed environment configurations (`app/config.py`) and reusable logger (`app/logger.py`).
*   [ ] **Sprint 2: Image Loader & Validator**
    *   Implement frame validation and robust geospatial data parsing (GeoTIFF/HDF5 support).
*   [ ] **Sprint 3: AI Interpolation Engine**
    *   Introduce core model interface, RIFE model adapter, and local CPU/GPU inference.
*   [ ] **Sprint 4: API & Web App Integration**
    *   FastAPI endpoints for upload and jobs, React-based visualization dashboard.
