<<<<<<< HEAD
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
=======
# FrameSat AI

# 🌍 The Problem

Satellite imagery is one of the most important sources of information for monitoring floods, cyclones, thunderstorms, wildfires, and other natural disasters. However, satellite observations are captured at fixed time intervals, often leaving significant temporal gaps between consecutive images.

During rapidly evolving events, critical changes can occur between these observations. Disaster analysts are forced to estimate what happened between two frames, making it difficult to accurately understand the progression of an event.

These missing observations can reduce situational awareness, delay decision-making, and limit the effectiveness of disaster response.

---

# 🎯 Purpose

FrameSat AI aims to bridge these temporal gaps using Artificial Intelligence.

The goal is to generate realistic intermediate satellite frames between two consecutive observations, allowing analysts to visualize how weather systems and disaster events evolve over time.

Beyond frame generation, the platform is designed to provide an intuitive workspace where users can upload imagery, explore satellite datasets, compare results, analyze metadata, and better understand the progression of natural disasters.

Our vision is to transform static satellite imagery into continuous, actionable visual intelligence.

---

# 💡 Our Solution

FrameSat AI combines modern AI models with a modular Earth Observation platform to create a complete disaster analysis workflow.

The platform enables users to:

- Upload two satellite images or select observations from supported datasets.
- Generate AI-interpolated intermediate frames using deep learning.
- Compare original and generated imagery side by side.
- Analyze metadata associated with each observation.
- Build smoother temporal timelines of evolving weather events.
- Export generated results for further research and reporting.

The system is designed using Clean Architecture and Domain-Driven Design principles, making it scalable, maintainable, and ready for future expansion.

Future versions will support multiple satellite datasets, additional interpolation models, climate analysis tools, temporal forecasting, and AI-assisted disaster intelligence.
---

## ✨ Features

### Current

- Modular Clean Architecture
- FastAPI Backend
- Next.js Frontend
- Domain-Driven Design
- Structured Logging
- Global Exception Handling
- Strongly Typed Configuration
- REST API
- Comprehensive Documentation

### Upcoming

- Image Upload Workflow
- SEVIR Dataset Integration
- Practical-RIFE Frame Interpolation
- Interactive Comparison Viewer
- AI-powered Timeline Generation
- Export Results
- Disaster Metadata Analysis
- Multi-model Support
- GPU Acceleration

---

# 🏗 Architecture

FrameSat AI follows a layered architecture inspired by **Clean Architecture** and **Domain-Driven Design (DDD)**.

```text
Presentation
        │
        ▼
Application
        │
        ▼
Domain
        ▲
        │
Infrastructure
```

### Layers

| Layer | Responsibility |
|---------|---------------|
| **Presentation** | REST API and HTTP endpoints |
| **Application** | Use cases and orchestration |
| **Domain** | Core business logic |
| **Infrastructure** | AI models, providers, storage |
| **Shared** | Configuration, exceptions, utilities |

---

# 📂 Project Structure

```text
FrameSat-AI/

├── backend/
│
│   ├── app/
│   │
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   ├── presentation/
│   ├── shared/
│   │
│   ├── main.py
│   ├── container.py
│   └── lifespan.py
│
├── frontend/
│
├── docs/
│
├── tests/
│
└── README.md
```

---

# 🚀 Technology Stack

## Backend

- FastAPI
- Python 3.11+
- Pydantic v2
- Uvicorn
- Practical-RIFE
- OpenCV
- Pillow
- NumPy

---

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- Framer Motion

---

## AI & Machine Learning

- PyTorch
- Practical-RIFE
- CUDA
- CPU Fallback

---

## Datasets

Planned support includes:

- SEVIR
- NOAA GOES
- INSAT
- Sentinel
- User Uploads

---

# ⚡ Getting Started

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/FrameSat-AI.git

cd FrameSat-AI
```

---

## Backend Setup

```bash
cd backend

python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run backend

```bash
uvicorn app.main:app --reload
```

Backend

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend

```
http://localhost:3000
```

---

# 🛣 Development Roadmap

## Sprint 1 ✅

- Project Foundation
- Clean Architecture
- Domain Layer
- Configuration
- Logging
- Exception Handling
- FastAPI Bootstrap

---

## Sprint 2 🚧

- Upload Provider
- SEVIR Provider
- Practical-RIFE Integration
- AI Pipeline
- First End-to-End Inference

---

## Sprint 3

- Interactive Workspace
- Timeline
- Before / After Viewer
- Metadata Panel
- Export

---

## Sprint 4

- Multi-model Support
- GPU Optimization
- Dataset Expansion
- Production Deployment

---

# 📖 Documentation

Project documentation includes:

- Software Requirements Specification (SRS)
- Architecture Decision Records (ADR)
- API Documentation
- Developer Guide
- User Guide

---

# 🎯 Vision

FrameSat AI aims to become a modern Earth Observation platform that assists researchers, meteorologists, and disaster analysts by generating high-quality temporal satellite imagery using AI.

The project is designed to evolve beyond frame interpolation into a comprehensive disaster intelligence and climate analysis platform.

---

# 🤝 Contributing

Contributions are welcome.

If you would like to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Nish**

Computer Science Engineering  


---

## ⭐ If you find this project interesting, consider giving it a star!
>>>>>>> d71e851b37affd913077e0006d49e55ebfc4c6bb
