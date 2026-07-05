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
