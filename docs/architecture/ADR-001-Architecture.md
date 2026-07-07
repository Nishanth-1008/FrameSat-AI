# ADR-001 — System Architecture

**Project:** FrameSat AI  
**Document ID:** ADR-001  
**Version:** 1.0.0  
**Status:** Accepted  
**Date:** July 2026  
**Authors:** Nishanth, ChatGPT Go.

---

# 1. Purpose

This Architecture Decision Record (ADR) defines the system architecture for **FrameSat AI**.

It serves as the single source of truth for every architectural decision made throughout the development lifecycle.

Every module, feature, API, component, dataset integration, and AI model must comply with the decisions documented in this ADR.

This document is intentionally technology-aware but implementation-independent.

It explains **why** decisions are made rather than describing implementation details.

Implementation is documented separately.

---

# 2. Document Scope

This ADR defines:

- Product vision
- Engineering philosophy
- System architecture
- Layer responsibilities
- High-level communication
- Core engineering principles
- Technology choices
- Dependency rules
- Architectural constraints

This document does **not** define:

- API endpoints
- UI wireframes
- Database schema
- Individual classes
- Function implementations

Those are covered by later ADRs.

---

# 3. Product Vision

## Vision Statement

FrameSat AI is an AI-powered satellite temporal intelligence platform that reconstructs missing observations between satellite acquisitions to improve temporal resolution for disaster monitoring and geospatial analysis.

Rather than simply generating an intermediate image, the platform provides analysts with a complete workflow for exploring temporal satellite imagery, generating AI-enhanced observations, comparing results, and exporting findings.

---

## Mission

Enable disaster analysts to observe rapidly evolving environmental events more effectively by reconstructing temporally missing satellite observations using modern deep learning models.

---

## Long-Term Vision

FrameSat AI should evolve into a modular Earth Observation AI platform capable of supporting multiple datasets, multiple interpolation models, and additional downstream geospatial intelligence tasks.

The architecture shall therefore prioritize extensibility over short-term optimization.

---

# 4. Product Goals

The system shall:

- Increase apparent temporal resolution of satellite imagery.
- Support multiple satellite datasets.
- Support user-uploaded satellite imagery.
- Provide a professional analysis interface.
- Maintain a modular AI pipeline.
- Allow replacement of interpolation models.
- Support future AI capabilities without redesigning the architecture.

---

# 5. Non-Goals

Version 1.0 intentionally excludes:

- Satellite forecasting
- Weather prediction
- Object detection
- Semantic segmentation
- Change detection
- Distributed inference
- Live satellite ingestion
- Multi-user collaboration
- Model training
- Cloud-native scaling

These capabilities remain future roadmap items.

---

# 6. Target Users

## Primary User

### Disaster Analyst

Responsibilities:

- Monitor evolving disasters
- Compare satellite observations
- Analyze temporal changes
- Generate reports

Primary expectations:

- Fast workflow
- Minimal technical complexity
- High visual quality
- Reliable outputs

---

## Secondary Users

- Remote sensing researchers
- Meteorologists
- Government agencies
- Students
- Demonstration audiences
- Hackathon judges

---

# 7. Core Problem

Satellite imagery is inherently discrete.

For example:

10:00

↓

10:10

↓

10:20

Rapidly changing events evolve continuously during these observation gaps.

Examples include:

- Cyclones
- Thunderstorms
- Floods
- Wildfires
- Dust storms
- Cloud evolution

These temporal gaps reduce situational awareness.

FrameSat AI addresses this limitation by reconstructing plausible intermediate observations.

---

# 8. Product Positioning

FrameSat AI is **not** an image editor.

FrameSat AI is **not** a computer vision demo.

FrameSat AI is a scientific decision-support platform.

Its purpose is to improve temporal understanding of Earth observation imagery.

Interpolation is the enabling technology—not the product itself.

---

# 9. Engineering Philosophy

The project shall follow the following engineering principles.

---

## Principle 1 — Modularity

Every subsystem must be independently replaceable.

No subsystem shall directly depend on implementation details of another subsystem.

Modules communicate only through documented interfaces.

---

## Principle 2 — Separation of Concerns

Each module has exactly one responsibility.

Examples:

Validation validates.

Preprocessing preprocesses.

Inference performs inference.

Visualization visualizes.

No module performs multiple unrelated responsibilities.

---

## Principle 3 — Provider Independence

The interpolation pipeline shall never depend on a specific dataset.

Instead, datasets expose a common provider interface.

Examples:

- Upload Provider
- SEVIR Provider
- NOAA Provider
- INSAT Provider

The AI pipeline consumes frames—not datasets.

---

## Principle 4 — Model Independence

The architecture shall never assume RIFE is permanent.

Today's implementation:

ECCV2022-RIFE

Possible future replacements:

- FILM
- IFRNet
- AMT
- Custom Transformer

Replacing the interpolation model must not require frontend changes.

---

## Principle 5 — Interface First

Every subsystem exposes interfaces before implementations.

Implementation details remain internal.

This minimizes coupling.

---

## Principle 6 — Testability

Every module shall be individually testable.

No module may require the full application to execute unit tests.

---

## Principle 7 — Observability

Every significant operation shall produce structured logs.

Important events include:

- Data loading
- Validation
- Inference
- Export
- Errors

Observability is considered a first-class architectural requirement.

---

## Principle 8 — Fail Gracefully

Unexpected failures shall never crash the application.

Every failure should:

- explain what happened
- explain why
- suggest recovery

Stack traces must never be shown to end users.

---

# 10. Architectural Drivers

The following constraints influenced the architecture.

## Functional Drivers

- Satellite image interpolation
- Dataset browsing
- Upload support
- Metrics reporting
- Visualization
- Export

---

## Quality Drivers

- Maintainability
- Extensibility
- Reliability
- Testability
- Usability
- Performance

---

## Business Drivers

- Bharatiya Antariksh Hackathon
- Portfolio quality
- Research extensibility
- Open-source readiness

---

# 11. High-Level System Overview

FrameSat AI consists of six major subsystems.

1. Presentation Layer
2. API Layer
3. Orchestration Layer
4. AI Processing Pipeline
5. Data Providers
6. Storage Layer

Each subsystem is isolated.

Communication always flows downward through clearly defined interfaces.

No subsystem bypasses architectural boundaries.

---

# 12. System Context

At the highest level, FrameSat AI interacts with three external entities.

## User

Provides:

- uploaded imagery
- dataset selections
- interpolation requests

Receives:

- generated imagery
- visualizations
- reports
- metrics

---

## Satellite Datasets

Examples:

- SEVIR
- NOAA GOES
- INSAT
- Sentinel

Datasets provide observations only.

They do not participate in processing.

---

## AI Model

The interpolation model is treated as an external computational dependency.

FrameSat AI prepares inputs, invokes inference, and processes outputs.

The model itself is never responsible for application logic.

---

# 13. High-Level Architecture

                +--------------------------------------+
                |          Presentation Layer          |
                |         (Next.js Frontend)           |
                +------------------+-------------------+
                                   |
                                   |
                                   ▼
                +--------------------------------------+
                |             API Layer                |
                |             (FastAPI)                |
                +------------------+-------------------+
                                   |
                                   ▼
                +--------------------------------------+
                |        Orchestration Layer           |
                +------------------+-------------------+
                                   |
          +------------------------+------------------------+
          |                         |                        |
          ▼                         ▼                        ▼
+------------------+     +--------------------+     +------------------+
| Data Providers   |     |   AI Pipeline      |     | Export Services  |
+------------------+     +--------------------+     +------------------+
          |                         |                        |
          +-------------------------+------------------------+
                                   |
                                   ▼
                     +-----------------------------+
                     |        Storage Layer        |
                     +-----------------------------+

---

# 14. Architecture Style

FrameSat AI adopts a **Layered Modular Monolith** architecture.

Reasons:

- Simpler than microservices
- Easier debugging
- Lower operational overhead
- Faster hackathon development
- Easier academic extension

The architecture intentionally avoids premature distribution.

Future service decomposition remains possible.

---

# 15. Architecture Principles Summary

The architecture shall satisfy the following goals:

- Modular
- Replaceable
- Observable
- Testable
- Extensible
- Dataset-independent
- Model-independent
- User-focused

These principles override implementation convenience.

Every future ADR shall conform to them.

---

# End of Part 1

# ADR-001 — System Architecture

# Part 2 — Repository Architecture, Layered Design & Dependency Rules

---

# 16. Repository Philosophy

FrameSat AI shall be developed as a **modular monorepo**.

A monorepo is chosen because:

- frontend and backend evolve together
- AI modules are tightly coupled with APIs
- documentation remains versioned with code
- easier onboarding
- easier CI/CD
- simpler dependency management

Although deployed as separate services, development occurs inside one repository.

---

# 17. Repository Layout

The repository shall follow this structure.

```text
FrameSat-AI/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── pull_request_template.md
│
├── backend/
│   │
│   ├── app/
│   │   ├── domain/
│   │   │   ├── entities/
│   │   │   ├── value_objects/
│   │   │   ├── enums/
│   │   │   └── interfaces/
│   │   │
│   │   ├── application/
│   │   │   ├── use_cases/
│   │   │   ├── dto/
│   │   │   └── services/
│   │   │
│   │   ├── infrastructure/
│   │   │   ├── providers/
│   │   │   ├── models/
│   │   │   ├── storage/
│   │   │   ├── logging/
│   │   │   └── persistence/
│   │   │
│   │   ├── presentation/
│   │   │   ├── routes/
│   │   │   ├── handlers/
│   │   │   ├── schemas/
│   │   │   └── api.py
│   │   │
│   │   ├── shared/
│   │   │   ├── config/
│   │   │   ├── exceptions/
│   │   │   └── utils/
│   │   │
│   │   ├── container.py
│   │   ├── lifespan.py
│   │   └── main.py
│   │
│   ├── tests/
│   │
│   └── requirements.txt
│
├── frontend/
│   │
│   ├── app/
│   │
│   ├── components/
│   │
│   ├── hooks/
│   │
│   ├── services/
│   │
│   ├── store/
│   │
│   ├── lib/
│   │
│   ├── types/
│   │
│   ├── styles/
│   │
│   ├── public/
│   │
│   └── package.json
│
├── datasets/
│
├── notebooks/
│
├── docs/
│
├── scripts/
│
├── experiments/
│
├── README.md
│
├── LICENSE
│
└── CONTRIBUTING.md
```

This structure is intentionally stable.

Folders should rarely change after Version 1.0.

---

# 18. Architectural Layers

The system is divided into six independent layers.

```text
Presentation Layer
↓
API Layer
↓
Application Layer
↓
AI Processing Layer
↓
Provider Layer
↓
Storage Layer
```

Each layer has clearly defined responsibilities.

Communication always flows downward.

---

# 19. Layer Responsibilities

---

## Layer 1 — Presentation Layer

Technology

- Next.js
- React
- Tailwind
- Framer Motion

Responsibilities

- Render UI
- Handle user interaction
- Display results
- Call backend APIs

Must NEVER

- load AI models
- preprocess images
- access datasets directly

---

## Layer 2 — API Layer

Technology:
- FastAPI
- Responsibilities
- Receive requests
- Validate requests
- Authenticate (future)
- Route requests
- Serialize responses

Must NEVER

- contain AI logic
- manipulate datasets
- perform interpolation

---

## Layer 3 — Application Layer

Responsibilities

Coordinate the system.

Acts as the conductor.

Example

```
User uploads image
↓
Validation
↓
Preprocessing
↓
Inference
↓
Postprocessing
↓
Response
```

This layer knows HOW modules interact.

It does NOT know HOW they work internally.

---

## Layer 4 — AI Processing Layer

Contains

```
Validator
↓
Preprocessor
↓
Interpolator
↓
Postprocessor
↓
Metrics
```

This layer performs all computational work.

It is independent of

- frontend
- FastAPI
- datasets

It only consumes tensors.

---

## Layer 5 — Provider Layer

Provides data.

Examples

```
Upload Provider

SEVIR Provider

NOAA Provider

INSAT Provider
```

Providers never perform AI.

They only provide frames and metadata.

---

## Layer 6 — Storage Layer

Contains:
- Datasets
- Outputs
- Logs
- Weights
- Cache
- Exports
No business logic exists here.

---

# 20. Dependency Rule

The most important architectural rule.

Dependencies always point downward.

Allowed

```
Frontend
↓
Backend
↓
Services
↓
Pipeline
↓
Providers
```

Forbidden

```
Provider
↓
Frontend
```

Forbidden

```
Metrics
↓
API
```

Forbidden

```
Interpolator
↓
Next.js
```

The dependency graph must never contain cycles.

---

# 21. Dependency Inversion

Every high-level module depends on interfaces.

Never implementations.

Wrong

```
SEVIRProvider
↓
Interpolator
```

Correct

```
DataProvider Interface
↓
Interpolator
```

Tomorrow

```
NOAAProvider
```

can replace

```
SEVIRProvider
```

without changing interpolation.

---

# 22. Repository Modules

The backend repository is organized into five primary architectural layers and entry modules inside the `app/` directory:

---

## domain/

Contains core business logic:
- `entities/`: Domain models (e.g., base entity, frame, dataset).
- `value_objects/`: Immutable data objects.
- `enums/`: Standard domain enums.
- `interfaces/`: Core model and provider interfaces (establishing dependency inversion).

---

## application/

Coordinates execution flow:
- `use_cases/`: Encapsulated business workflow/use cases.
- `dto/`: Data Transfer Objects for passing data across layers.
- `services/`: Orchestration services that call domain models and infrastructure.

---

## infrastructure/

Implements technical details and interfaces:
- `providers/`: Datasets and external data providers.
- `models/`: Wrappers for deep learning/AI interpolation models.
- `storage/`: Local and cloud file storage adapters.
- `logging/`: Application logger configuration.
- `persistence/`: Database or cache storage mapping.

---

## presentation/

Exposes functionality to clients:
- `routes/`: FastAPI API endpoints/routers.
- `handlers/`: Exception and error handlers.
- `schemas/`: Request and response validation schemas (Pydantic models).
- `api.py`: Core API router configuration.

---

## shared/

Common utilities and components:
- `config/`: Application settings and environment configuration.
- `exceptions/`: Cross-layer exception definitions.
- `utils/`: Reusable helper utilities.

---

## Entry Points

Application bootstrap files:
- `container.py`: Composition root and dependency injector.
- `lifespan.py`: Startup and shutdown hooks.
- `main.py`: FastAPI application entrypoint.

---

# 23. Frontend Modules

The frontend follows feature-based organization.

```
app/

components/

hooks/

store/

services/

lib/

types/
```

Responsibilities

---

app/

Routing

Layouts

Pages

---

components/

Reusable UI

---

hooks/

Business interaction

---

services/

Backend communication

---

store/

Application state

---

types/

Shared interfaces

---

lib/

Utilities

Theme

Constants

Helpers

---

# 24. Data Ownership

Every layer owns its own data.

Frontend

owns

```
UI State
```

Backend

owns

```
Business State
```

Providers

own

```
Dataset State
```

Models

own

```
Tensor State
```

No layer may mutate another layer's internal state.

---

# 25. Communication Rules

Communication occurs only through contracts.

Frontend

↓

REST API

Backend

↓

Services

Services

↓

Interfaces

Interfaces

↓

Implementations

No module calls another module directly if an interface exists.

---

# 26. Error Propagation

Errors move upward.

```
Provider
↓
Service
↓
API
↓
Frontend
```

Each layer adds context.

Example

Provider

```
File missing
```

↓

Service

```
Dataset sample missing
```

↓

API

```
404 Dataset Not Found
```

↓

Frontend

```
Selected dataset sample does not exist.
```

No raw exceptions reach users.

---

# 27. Logging Boundaries

Every layer logs only its own work.

Provider

logs loading.

Pipeline

logs inference.

API

logs requests.

Frontend

logs UI events only.

No duplicate logging.

---

# 28. Configuration Strategy

Every configurable value originates from one source.

```
Environment Variables
↓
Config
↓
Dependency Injection
↓
Application
```

Hardcoded values are prohibited except for true constants.

---

# 29. Folder Ownership Rules

Every folder has one owner.

Example

providers/

owns datasets.

core/

owns computation.

models/

owns AI.

services/

owns orchestration.

api/

owns HTTP.

No module crosses ownership boundaries.

---

# 30. Repository Stability Rules

The following folders are considered stable.

```
backend/

frontend/

docs/

datasets/

tests/
```

They should not be renamed after Version 1.0.

Changing the root architecture requires a new ADR.

---

# 31. Architecture Constraints

The following constraints are mandatory.

- No circular dependencies.
- No AI code in routes.
- No dataset code in frontend.
- No UI logic in backend.
- No provider-specific logic inside AI.
- No hardcoded dataset paths.
- No duplicated preprocessing.
- No business logic inside React components.

---

# 32. Architectural Summary

The repository is intentionally designed around clear ownership boundaries.

Every subsystem performs one responsibility.

Communication always follows documented contracts.

The architecture prioritizes:

- maintainability
- extensibility
- testability
- readability
- replaceability

These principles take precedence over implementation convenience.

---

# End of Part 2

# ADR-001 — System Architecture

# Part 3 — Backend Architecture

---

# 33. Backend Philosophy

The backend is the central nervous system of FrameSat AI.

Its responsibility is **not** to perform AI.

Its responsibility is to coordinate AI.

The backend acts as an orchestration platform between:

- Frontend
- Dataset Providers
- AI Pipeline
- Export Services
- Storage

The backend owns the application's business logic.

---

# 34. Backend Goals

The backend shall:

- expose stable REST APIs
- orchestrate the AI pipeline
- manage datasets
- validate requests
- handle errors
- log operations
- expose runtime metrics
- remain independent of frontend implementation

---

# 35. Architectural Style

The backend follows a layered architecture.

```text
                FastAPI Routes
                      │
                      ▼
              Application Services
                      │
                      ▼
              AI Processing Pipeline
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
 Dataset Providers               AI Models
      │                               │
      └───────────────┬───────────────┘
                      ▼
                   Storage
```

No layer skips another.

---

# 36. Backend Directory Structure

```text
backend/
│
├── app/
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── base_entity.py
│   │   │   ├── dataset.py
│   │   │   ├── frame.py
│   │   │   ├── frame_pair.py
│   │   │   ├── frame_sequence.py
│   │   │   ├── interpolation_result.py
│   │   │   └── provider.py
│   │   ├── value_objects/
│   │   │   ├── dimensions.py
│   │   │   ├── metadata.py
│   │   │   └── timestamp.py
│   │   ├── enums/
│   │   │   ├── dataset_type.py
│   │   │   ├── device_type.py
│   │   │   ├── image_format.py
│   │   │   ├── model_type.py
│   │   │   └── provider_type.py
│   │   └── interfaces/
│   │       ├── models/
│   │       │   └── interpolation_model.py
│   │       └── providers/
│   │           └── data_provider.py
│   │
│   ├── application/
│   │   ├── use_cases/
│   │   ├── dto/
│   │   └── services/
│   │
│   ├── infrastructure/
│   │   ├── providers/
│   │   ├── models/
│   │   ├── storage/
│   │   ├── logging/
│   │   │   └── logger.py
│   │   └── persistence/
│   │
│   ├── presentation/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── root.py
│   │   │   └── system.py
│   │   ├── handlers/
│   │   │   └── exceptions.py
│   │   ├── schemas/
│   │   └── api.py
│   │
│   ├── shared/
│   │   ├── config/
│   │   │   └── settings.py
│   │   ├── exceptions/
│   │   │   ├── base.py
│   │   │   ├── configuration.py
│   │   │   ├── domain.py
│   │   │   ├── inference.py
│   │   │   └── provider.py
│   │   └── utils/
│   │
│   ├── container.py
│   ├── lifespan.py
│   └── main.py
│
├── tests/
└── requirements.txt
```

---

# 37. Request Lifecycle

Every request follows the exact same lifecycle.

```text
HTTP Request
↓
Validation
↓
Authentication (future)
↓
Service
↓
Provider
↓
Preprocessing
↓
AI Model
↓
Postprocessing
↓
Metrics
↓
Response
```

No shortcuts.

---

# 38. Backend Responsibilities

The backend owns:

- request validation
- orchestration
- provider selection
- model selection
- error handling
- logging
- metrics
- export

The backend does **not** own:

- UI rendering
- visualization
- animations
- frontend state

---

# 39. Service Layer

The Service Layer contains the application's business logic.

Each service performs one responsibility.

```text
InterpolationService

DatasetService

ExportService

MetricsService

HealthService

SystemService
```

Services communicate with:

- providers
- models
- core modules

Never directly with React.

---

# 40. InterpolationService

The heart of the backend.

Responsibilities:

- receive interpolation request
- choose provider
- load frames
- validate input
- preprocess
- invoke model
- postprocess
- save output
- compute metrics
- return response

This service owns the complete interpolation workflow.

---

# 41. DatasetService

Responsibilities:

- discover datasets
- browse events
- load metadata
- retrieve frames
- expose dataset information

The DatasetService never performs AI inference.

---

# 42. ExportService

Responsibilities:

- save PNG
- generate GIF
- package ZIP
- create JSON reports
- generate comparison images

Future:

- PDF reports
- GeoTIFF export

---

# 43. MetricsService

Responsibilities:

- runtime metrics
- inference latency
- PSNR
- SSIM
- LPIPS
- memory usage
- GPU statistics

No visualization.

Only computation.

---

# 44. HealthService

Responsibilities:

Return application status.

Checks:

- model loaded
- GPU available
- providers available
- datasets accessible
- storage writable

Used by:

GET /health

---

# 45. SystemService

Provides system information.

Examples:

- application version
- model name
- device
- backend version
- active provider
- runtime configuration

Used by:

GET /system

---

# 46. API Layer

The API Layer contains no business logic.

Routes perform only:

- request parsing
- validation
- dependency injection
- service invocation
- response serialization

Never:

- preprocess images
- run inference
- access datasets directly

---

# 47. Dependency Injection

Every service is injected.

Example:

```text
Route
↓
InterpolationService
↓
Provider
↓
Model
```

Routes never instantiate services manually.

This improves:

- testing
- replacement
- mocking

---

# 48. Provider Resolution

The backend chooses the correct provider.

Example:

```text
Request
↓
source = upload
↓
UploadProvider
```

or

```text
Request
↓
source = sevir
↓
SEVIRProvider
```

The frontend never selects implementation classes.

Only source types.

---

# 49. Model Resolution

Current:

```text
RIFE
```

Future:

```text
FILM

IFRNet

AMT
```

Selection occurs through a Model Registry.

The backend asks:

```text
ModelRegistry
↓
Active Model
```

Never:

```python
RIFEWrapper()
```

inside services.

---

# 50. Request Validation

Validation occurs in multiple stages.

Stage 1

API schema validation.

Stage 2

Business validation.

Stage 3

Provider validation.

Stage 4

Image validation.

Stage 5

Pipeline validation.

Each stage produces meaningful errors.

---

# 51. Error Handling

Every backend error becomes a structured response.

Example:

```json
{
  "error": {
    "code": "IMAGE_SIZE_MISMATCH",
    "message": "Input images have different dimensions.",
    "hint": "Upload images with matching width and height."
  }
}
```

Never expose Python stack traces.

---

# 52. Logging Strategy

Every request receives a Request ID.

Logs include:

- request ID
- timestamp
- endpoint
- provider
- model
- runtime
- status

Example:

```text
[2026-07-05 14:30:01]

Request: 7f31c

Provider: SEVIR

Model: RIFE

Runtime: 182 ms

Status: SUCCESS
```

---

# 53. Backend State

The backend is largely stateless.

Persistent state:

- datasets
- outputs
- cache
- logs

Transient state:

- tensors
- inference pipeline
- metrics

No user session data is required for Version 1.

---

# 54. Background Tasks

Future operations that should run asynchronously:

- GIF generation
- report generation
- dataset indexing
- cache cleanup
- metric aggregation

Inference itself remains synchronous in Version 1.

---

# 55. Performance Principles

Backend performance priorities:

1. Low latency
2. Deterministic behavior
3. Stable memory usage
4. Predictable inference
5. Graceful degradation

Performance should never compromise correctness.

---

# 56. Backend Security

The backend must:

- validate uploads
- limit file size
- reject executable files
- sanitize filenames
- isolate temporary files
- never trust client metadata

Future:

- authentication
- authorization
- rate limiting

---

# 57. Backend Extensibility

Adding a new provider should require:

- one provider class
- one registration entry

Nothing else.

Adding a new AI model should require:

- one wrapper
- one registry entry

Nothing else.

The backend must remain closed for modification but open for extension.

---

# 58. Backend Summary

The backend acts as an orchestration platform.

It owns:

- workflow
- services
- APIs
- providers
- model coordination

It does **not** own:

- UI
- presentation
- visualization

Every backend component communicates through stable interfaces and follows strict dependency boundaries.

---

# End of Part 3

# ADR-001 — System Architecture

# Part 4 — Frontend Architecture

---

# 59. Frontend Philosophy

The frontend is not merely a graphical interface.

It is the user's primary workspace for interacting with FrameSat AI.

The frontend shall:

- guide the analyst through the workflow,
- visualize complex geospatial information,
- simplify AI interactions,
- present system status,
- communicate errors clearly,
- remain responsive and intuitive.

The frontend never performs AI inference.

Its responsibility is user experience.

---

# 60. Frontend Goals

The frontend shall:

- provide a professional analytical workspace
- minimize user confusion
- expose only necessary controls
- remain responsive on desktop and laptop
- support future mobile viewing
- integrate seamlessly with backend APIs
- provide clear feedback during long-running operations

---

# 61. Architectural Style

FrameSat AI adopts a **Feature-Oriented Frontend Architecture**.

Instead of organizing by file type only:

```
components/

hooks/

pages/
```

the application is organized around features.

Example:

```
Workspace
↓
Upload
↓
Dataset Browser
↓
Interpolation
↓
Viewer
↓
Export
```

Each feature owns its own components, hooks, services and state.

---

# 62. Technology Stack

Framework

- Next.js (App Router)

Language

- TypeScript

Styling

- Tailwind CSS

Animation

- Framer Motion

State

- Zustand

Server State

- TanStack Query

Icons

- Lucide React

Comparison Viewer

- React Compare Slider

Notifications

- Sonner

Charts (Future)

- Recharts

---

# 63. Frontend Directory Structure

```
frontend/

app/

components/

features/

hooks/

services/

store/

types/

lib/

styles/

public/

assets/
```

Every folder has a single responsibility.

---

# 64. App Router Structure

```
/

Landing Page

/workspace

Main Application

/settings

Application Settings

/docs

Documentation

/about

Project Information

/examples

Sample Demonstrations

/not-found

404
```

Future

```
/history

/profile

/admin
```

---

# 65. Workspace Layout

The Workspace is the heart of the application.

```
+---------------------------------------------------------------+
| Header                                                        |
+-----------+---------------------------------------+-----------+
| Sidebar   |                                       | Inspector |
|           |            Main Viewer                |           |
|           |                                       |           |
|           |                                       |           |
+-----------+---------------------------------------+-----------+
| Bottom Timeline / Status Bar                                  |
+---------------------------------------------------------------+
```

The layout remains consistent across all workflows.

---

# 66. Major UI Regions

## Header

Responsibilities

- Branding
- Backend status
- Model information
- Device information
- Theme toggle
- Settings

Never contains workflow controls.

---

## Sidebar

Responsibilities

- Data source selection
- Upload
- Dataset browser
- Generation controls
- Export controls

The sidebar is task-oriented.

---

## Main Viewer

Displays

- Frame A
- Generated Frame
- Frame B
- Comparison slider
- Difference map
- Zoom
- Pan

Only visualization.

---

## Inspector Panel

Displays

- Runtime
- Resolution
- Dataset
- Event ID
- Timestamp
- Model
- Device
- Metrics
- Warnings

The inspector is read-only.

---

## Timeline

Displays

```
Frame A
↓
Generated Frame
↓
Frame B
```

Future versions may support full sequence playback.

---

# 67. Component Hierarchy

```
App
↓
Workspace
↓
Header
↓
Sidebar
↓
Viewer
↓
Inspector
↓
Timeline
```

Each high-level component owns smaller reusable components.

---

# 68. Component Design Principles

Every component should satisfy:

- single responsibility
- reusable
- typed
- testable
- stateless when possible

Large components must be composed from smaller components.

---

# 69. State Management Strategy

Frontend state is divided into three categories.

---

## Local State

Managed with React.

Examples

- modal open
- dropdown expanded
- hover state

---

## Global State

Managed with Zustand.

Examples

- selected provider
- uploaded frames
- interpolation result
- application settings

---

## Server State

Managed with TanStack Query.

Examples

- system status
- datasets
- metrics
- backend configuration

Server state must never be duplicated.

---

# 70. Store Organization

```
store/

useSystemStore

useUploadStore

useDatasetStore

useInterpolationStore

useViewerStore

useSettingsStore

useMetricsStore
```

Each store owns one domain.

Stores never depend on each other directly.

---

# 71. Services

Services communicate with backend APIs.

Example

```
SystemService

DatasetService

InterpolationService

ExportService

MetricsService
```

Services contain HTTP logic only.

No UI logic.

---

# 72. Hooks

Hooks coordinate UI and services.

Example

```
useInterpolation()
↓
calls
InterpolationService
↓
updates
InterpolationStore
```

Hooks never render UI.

---

# 73. Design System

The design system is mandatory.

All colors

spacing

radius

animations

typography

icons

come from one source.

No hardcoded design values.

---

# 74. Motion Philosophy

Animation exists to improve understanding.

Never for decoration.

Examples

Good

- upload progress
- interpolation stages
- sidebar transitions
- page transitions

Avoid

- excessive bouncing
- distracting effects
- meaningless animations

---

# 75. Loading States

Every asynchronous action must have a loading state.

Examples:
- Uploading
- Reading Dataset
- Loading Event
- Generating Frame
- Exporting
Never leave the user waiting without feedback.

---

# 76. Error States

Every failure produces a user-friendly message.

Example

Bad

```
HTTP 500
```

Good

```
Interpolation failed.

The selected images could not be processed.

Please verify both images belong to the same sequence.
```

---

# 77. Empty States

Every page has an intentional empty state.

Example

Workspace

```
No data selected.

Choose a dataset

or

Upload two images

to begin.
```

Empty states educate users.

---

# 78. Accessibility

The frontend shall support:

- keyboard navigation
- focus indicators
- semantic HTML
- ARIA labels
- high contrast
- scalable typography

Accessibility is not optional.

---

# 79. Responsive Design

Primary target

Desktop

Secondary

Laptop

Future

Tablet

Mobile phones are not a Version 1 priority but layouts should degrade gracefully.

---

# 80. Communication Flow

The frontend communicates only with REST APIs.

```
Component
↓
Hook
↓
Service
↓
API
↓
Backend
```

Components never perform fetch requests directly.

---

# 81. Frontend Performance

Guidelines

- Lazy load large components
- Memoize expensive renders
- Virtualize long lists
- Optimize images
- Avoid unnecessary re-renders

Performance is part of UX.

---

# 82. Frontend Security

The frontend must never:

- store secrets
- expose API keys
- trust user input
- execute uploaded files

Validation always occurs again on the backend.

---

# 83. Frontend Extensibility

Adding a new feature should require:

- one feature folder
- one service
- one store (if needed)

Existing features should remain untouched.

---

# 84. Frontend Summary

The frontend is designed as a modular analytical workspace.

Its responsibilities are:

- user interaction
- visualization
- workflow guidance
- state management
- backend communication

It intentionally contains:

- no AI logic
- no preprocessing
- no dataset parsing
- no business logic

Those responsibilities remain within the backend.

---

# End of Part 4

# ADR-001 — System Architecture

# Part 5 — AI Processing Architecture

---

# 85. Purpose

The AI Processing Layer is responsible for transforming raw satellite observations into AI-generated intermediate observations.

It represents the computational core of FrameSat AI.

Unlike the Backend Layer, which orchestrates requests, the AI Processing Layer performs numerical computation.

Its responsibilities include:

- validation
- preprocessing
- inference
- postprocessing
- metric computation

No UI or HTTP logic exists in this layer.

---

# 86. Design Goals

The AI Processing Layer shall:

- remain model-independent
- remain dataset-independent
- be deterministic
- be testable
- support CPU and GPU execution
- allow future model replacement
- support multiple interpolation algorithms

---

# 87. AI Pipeline Overview

Every interpolation request follows the same pipeline.

```
Input
│
▼
Validation
│
▼
Metadata Extraction
│
▼
Preprocessing
│
▼
Tensor Preparation
│
▼
Interpolation Model
│
▼
Postprocessing
│
▼
Quality Metrics
│
▼
Output Packaging
│
▼
Response
```

No stage may be skipped.

---

# 88. Processing Philosophy

Every stage performs exactly one responsibility.

```
Validation
↓
Preprocessing
↓
Inference
↓
Postprocessing
↓
Metrics
```

This separation makes debugging significantly easier.

---

# 89. Pipeline Components

The AI Pipeline consists of five primary modules.

```
Validator
↓
Preprocessor
↓
Interpolator
↓
Postprocessor
↓
Metrics
```

Each module is independently replaceable.

---

# 90. Validator

Purpose

Ensure inputs are safe and compatible before inference.

Responsibilities

- verify file integrity
- verify image dimensions
- verify channels
- verify datatype
- verify metadata
- verify provider output

Input

```

FramePair

```

Output

```

ValidatedFramePair

```

Possible Errors

- file missing
- corrupted image
- unsupported format
- mismatched dimensions
- invalid channels

---

# 91. Metadata Extraction

Before preprocessing begins, metadata is extracted.

Common metadata

- filename
- width
- height
- channels
- timestamp
- provider
- sensor

Optional metadata

- CRS
- transform
- band
- spatial resolution
- acquisition angle

Metadata remains immutable throughout the pipeline.

---

# 92. Preprocessor

Purpose

Transform provider output into model-ready tensors.

Responsibilities

- resize
- normalize
- channel conversion
- datatype conversion
- tensor conversion
- device transfer

Input

```

ValidatedFramePair

```

Output

```

TensorPair

```

---

# 93. Standardization Rules

Every interpolation model receives identical input.

Standard

```

float32

[B,C,H,W]

Range [0,1]

RGB

```

No model-specific preprocessing is allowed outside the model wrapper.

---

# 94. Tensor Preparation

The pipeline converts images into tensors.

Workflow

```
Image
↓
NumPy
↓
Tensor
↓
Normalize
↓
Batch
↓
Device
```

Tensor preparation remains model-independent.

---

# 95. Model Interface

Every interpolation model implements the same interface.

```

BaseInterpolator

```

Required methods

```

load()

interpolate()

warmup()

shutdown()

metadata()

```

Every future model must implement these methods.

---

# 96. Current Model

Version 1 uses

ECCV2022-RIFE

Reasons

- mature
- stable
- lightweight
- fast
- strong visual quality

The architecture does not depend on RIFE.

---

# 97. Future Models

Possible future implementations

- FILM
- IFRNet
- AMT
- XVFI
- Custom Transformer

No pipeline changes required.

Only a new wrapper.

---

# 98. Model Registry

Models are never instantiated directly.

Instead

```
ModelRegistry
↓
Active Model
```

Configuration selects the active implementation.

Example

```
RIFE
↓
Registry
↓
Interpolator
```

---

# 99. Inference Stage

Responsibilities

- receive tensors
- perform interpolation
- return predicted tensor

No preprocessing.

No saving.

No metrics.

Only inference.

---

# 100. Multi-Step Interpolation

Version 1

```
A
↓
0.5
↓
B
```

Future

```
A
↓
0.25
↓
0.50
↓
0.75
↓
B
```

The pipeline already supports arbitrary timestep values.

---

# 101. Device Management

The pipeline automatically selects execution device.

Priority

```
CUDA
↓
MPS
↓
CPU
```

Device selection occurs once during startup.

Individual modules never choose devices.

---

# 102. Postprocessor

Purpose

Convert model output into user-ready imagery.

Responsibilities

- denormalization
- clipping
- datatype conversion
- image reconstruction
- metadata restoration

Input

```

Prediction Tensor

```

Output

```

Generated Frame

```

---

# 103. Metadata Preservation

Whenever possible

preserve

- timestamps
- provider information
- sensor information
- projection metadata

Interpolation should never silently discard metadata.

---

# 104. Output Packaging

The AI pipeline returns a structured object.

```

InterpolationResult

```

Contains

- generated frame
- runtime
- model
- device
- metadata
- warnings

No file paths.

Saving occurs later.

---

# 105. Metrics

Metrics are computed after inference.

Current

- runtime
- latency
- resolution
- memory

Future

- PSNR
- SSIM
- LPIPS
- MAE
- Edge Preservation

Metrics never modify outputs.

---

# 106. Quality Flags

Instead of binary success

the pipeline produces quality indicators.

Example

```

Alignment Warning

Fast Motion

Possible Artifacts

Large Temporal Gap

```

These help analysts interpret results.

---

# 107. Error Recovery

Failures occur at individual stages.

Example

```
Validation
↓
Fail
↓
Stop
```

or

```
Inference
↓
GPU Error
↓
Retry CPU
↓
Continue
```

Every stage should fail independently.

---

# 108. Performance Strategy

Goals

- minimal copies
- lazy loading
- reuse tensors
- model warmup
- avoid repeated allocations

Performance optimizations must never change numerical output.

---

# 109. AI Pipeline State

The pipeline is stateless.

Persistent state

- loaded model
- configuration

Transient state

- tensors
- predictions
- metrics

No inference history is stored inside the pipeline.

---

# 110. Logging

Each stage logs

```
Start
↓
End
↓
Duration
↓
Status
```

Example

```

[Pipeline]

Validation

14 ms

SUCCESS

```

---

# 111. Thread Safety

The AI Pipeline must support concurrent requests in the future.

Version 1 processes one request at a time.

The design avoids global mutable state to simplify future parallel execution.

---

# 112. Extensibility

Adding a new preprocessing stage should require

```
Pipeline
↓
Register Stage
↓
Done
```

Adding a new model should require

```
Model Wrapper
↓
Registry
↓
Done
```

Adding a new metric should require

```
Metric Class
↓
Registry
↓
Done
```

The pipeline remains closed for modification but open for extension.

---

# 113. AI Pipeline Summary

The AI Processing Layer is completely isolated from

- HTTP
- React
- datasets
- providers
- storage

It consumes standardized tensors and produces standardized predictions.

This isolation guarantees

- portability
- maintainability
- testability
- future model replacement

without changing the surrounding application.

---

# End of Part 5

# ADR-001 — System Architecture

# Part 6 — Data Provider Architecture

---

# 114. Purpose

The Data Provider Layer abstracts every data source used by FrameSat AI.

The AI Pipeline must never know:

- where data originated
- how data is stored
- how metadata is organized
- how frames are retrieved

Instead, every data source exposes the exact same interface.

This allows FrameSat AI to support:

- uploaded imagery
- SEVIR
- NOAA GOES
- INSAT
- Sentinel
- future datasets

without modifying the AI Pipeline.

---

# 115. Design Goals

The Provider Layer shall:

- isolate dataset-specific logic
- expose a common interface
- support local and remote datasets
- expose metadata consistently
- simplify adding new datasets
- remain independent of AI models

---

# 116. Why Providers?

Without providers

```
InterpolationService
↓
if dataset == "SEVIR"
↓
if dataset == "GOES"
↓
if dataset == "INSAT"
↓
...
```

This quickly becomes impossible to maintain.

Instead

```
InterpolationService
↓
DataProvider Interface
↓
Concrete Provider
```

The application never knows which provider is being used.

---

# 117. Provider Architecture

```
                    DataProvider
                         ▲
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
 UploadProvider     SEVIRProvider     NOAAProvider
                                             │
                                             ▼
                                       INSATProvider
```

Every provider inherits from DataProvider.

---

# 118. DataProvider Interface

Every provider must implement:

```
initialize()

discover()

list()

metadata()

load()

load_pair()

load_sequence()

validate()

close()
```

No additional required methods.

Optional methods are allowed.

---

# 119. Provider Responsibilities

A provider is responsible for:

- opening datasets
- locating samples
- retrieving frames
- exposing metadata
- validating dataset integrity
- closing resources

A provider must never:

- interpolate
- normalize tensors
- compute metrics
- perform inference

---

# 120. UploadProvider

Purpose

Support user uploaded imagery.

Supported formats

- PNG
- JPG
- JPEG
- TIFF
- GeoTIFF
- NumPy

Responsibilities

- validate uploads
- read metadata
- return FramePair

UploadProvider behaves like any other dataset.

---

# 121. SEVIRProvider

Purpose

Provide access to SEVIR datasets.

Supported storage

- Zarr

Responsibilities

- discover events
- browse sequences
- retrieve frames
- retrieve timestamps
- expose labels
- expose metadata

The provider understands SEVIR.

No other module does.

---

# 122. Future Providers

The architecture already supports:

GOES

INSAT

Sentinel

Himawari

ERA5

Custom datasets

Adding one requires only:

```
New Provider
↓
Registry
↓
Done
```

---

# 123. Provider Registry

Providers are never instantiated manually.

Instead

```
ProviderRegistry
↓
Available Providers
↓
Selected Provider
```

The registry owns provider discovery.

---

# 124. Provider Factory

Selection occurs through a factory.

Example

```
source="upload"
↓
UploadProvider
```

```
source="sevir"
↓
SEVIRProvider
```

No service creates providers directly.

---

# 125. Dataset Discovery

Every provider supports discovery.

Example

```
datasets/
↓
discover()
↓
Dataset List
```

Returned information

- name
- description
- type
- provider
- availability

---

# 126. Sample Discovery

Datasets expose samples.

Example

```
SEVIR
↓
Events
↓
Event IDs
```

Returned

```
SampleSummary
```

Contains

- ID
- timestamp
- event type
- duration
- frame count

---

# 127. Metadata Contract

Every provider returns metadata in a common format.

Required

```
Provider

Dataset

Sample ID

Timestamp

Width

Height

Channels

Datatype
```

Optional

```
CRS

Spatial Resolution

Projection

Satellite

Sensor

Bands
```

Missing fields become

```
None
```

Never omitted.

---

# 128. Frame Contract

Every provider returns the same object.

```
Frame
```

Contains

```
Image

Metadata

Timestamp

Provider
```

No raw NumPy arrays are returned directly to services.

---

# 129. Frame Pair Contract

Interpolation always consumes

```
FramePair
```

Contains

```
Frame A

Frame B

Metadata
```

The pipeline never receives independent images.

---

# 130. Sequence Contract

Future video interpolation requires

```
FrameSequence
```

Contains

```
Frames

Metadata

Timeline
```

Version 1 primarily uses FramePair.

---

# 131. Validation

Providers validate:

- sample exists
- frame exists
- timestamps valid
- metadata complete

They do not validate tensors.

Tensor validation belongs to the pipeline.

---

# 132. Caching

Providers may cache

- metadata
- frame indices
- dataset handles

Providers must never cache:

- inference outputs
- tensors
- metrics

---

# 133. Large Dataset Strategy

Datasets such as SEVIR are too large to load completely.

Providers therefore implement:

```
Lazy Loading
```

Workflow

```
Dataset
↓
Open Handle
↓
Load Event
↓
Load Frame
↓
Return
```

Entire datasets are never loaded into memory.

---

# 134. Resource Management

Every provider owns its resources.

Resources include

- file handles
- Zarr stores
- memory maps
- caches

Every provider must implement

```
close()
```

to release resources.

---

# 135. Error Handling

Provider errors remain provider-specific.

Example

```
Dataset Missing
↓
ProviderError
↓
Service
↓
DatasetUnavailable
↓
API
↓
404
```

Errors become progressively more user-friendly.

---

# 136. Logging

Providers log

- dataset opened
- sample selected
- frame loaded
- metadata extracted
- dataset closed

No inference logging.

---

# 137. Performance Goals

Providers should

- load only required frames
- minimize disk reads
- reuse open dataset handles
- cache metadata

Avoid

- repeated dataset opening
- unnecessary image decoding
- loading unused frames

---

# 138. Thread Safety

Each provider instance owns its own state.

Shared mutable dataset state is prohibited.

Future parallel inference should require no provider redesign.

---

# 139. Extensibility

Adding a dataset requires

1.

Create Provider

↓

2.

Implement Interface

↓

3.

Register Provider

↓

Done

No existing provider should be modified.

---

# 140. Provider Summary

The Provider Layer isolates every dataset-specific concern.

It guarantees that the AI Pipeline always receives standardized data regardless of source.

This architecture enables FrameSat AI to evolve from supporting SEVIR and uploaded imagery today to supporting additional satellite missions in the future without changing the interpolation engine.

---

# End of Part 6

# ADR-001 — System Architecture

# Part 7 — Communication Architecture, API Contracts & State Management

---

# 141. Purpose

The Communication Architecture defines how information flows throughout FrameSat AI.

Every subsystem communicates through documented contracts.

Direct communication between unrelated modules is prohibited.

This guarantees:

- loose coupling
- scalability
- maintainability
- testability
- replaceability

---

# 142. Communication Principles

FrameSat AI follows five communication principles.

## Principle 1

Communication is directional.

Information always moves through predefined layers.

Never sideways.

---

## Principle 2

Communication occurs only through contracts.

Never through implementation details.

---

## Principle 3

Every request produces one response.

Long-running tasks may additionally produce progress events.

---

## Principle 4

Every message has a defined schema.

No anonymous dictionaries.

No loosely typed payloads.

---

## Principle 5

Communication is stateless.

Each request contains everything required to process it.

---

# 143. Overall Communication Flow

```
User
↓
Frontend
↓
REST API
↓
Backend Service
↓
Provider
↓
AI Pipeline
↓
Provider
↓
Service
↓
REST Response
↓
Frontend
↓
User
```

The frontend never communicates directly with:

- datasets
- models
- providers
- storage

---

# 144. Communication Layers

The application consists of six communication boundaries.

```
Presentation
↓
Transport
↓
Application
↓
Domain
↓
Infrastructure
↓
Storage
```

Each boundary owns its own contracts.

---

# 145. REST API Philosophy

The backend exposes a REST API.

REST was selected because:

- simplicity
- browser compatibility
- debugging ease
- documentation support
- future SDK generation

GraphQL is intentionally excluded from Version 1.

---

# 146. API Versioning

Every endpoint belongs to a version.

Example

```
/api/v1/system

/api/v1/interpolate

/api/v1/providers

/api/v1/datasets
```

Future versions

```
/api/v2/
```

must remain backward compatible whenever possible.

---

# 147. Endpoint Categories

The API is organized by domain.

```
System

Providers

Datasets

Interpolation

Metrics

Exports

Health
```

Each category owns its own routes.

---

# 148. System Endpoints

Examples

```
GET /system

GET /health

GET /version
```

Purpose

Provide application information.

---

# 149. Provider Endpoints

Examples

```
GET /providers

GET /providers/{id}

GET /providers/status
```

Purpose

Discover available providers.

---

# 150. Dataset Endpoints

Examples

```
GET /datasets

GET /datasets/{provider}

GET /datasets/{provider}/{sample}

GET /datasets/{provider}/{sample}/frames
```

Purpose

Browse datasets without loading entire datasets.

---

# 151. Interpolation Endpoints

Examples

```
POST /interpolate

POST /interpolate/upload

POST /interpolate/sevir
```

Version 1 may expose a single endpoint.

Internally all requests use the same pipeline.

---

# 152. Export Endpoints

Examples

```
GET /exports

GET /download/{id}

POST /export/gif

POST /export/report
```

Future

GeoTIFF export.

---

# 153. Metrics Endpoints

Examples

```
GET /metrics

GET /metrics/runtime

GET /metrics/history
```

Metrics remain read-only.

---

# 154. Health Endpoints

Examples

```
GET /health

GET /health/model

GET /health/provider
```

Used by

- frontend
- deployment
- monitoring

---

# 155. Request Lifecycle

Every request follows exactly the same lifecycle.

```
Receive Request
↓
Validate
↓
Resolve Provider
↓
Resolve Model
↓
Execute Service
↓
Return Result
↓
Serialize
↓
Respond
```

No endpoint bypasses this lifecycle.

---

# 156. Request Models

Every request uses explicit schemas.

Example

```
InterpolationRequest
↓
ProviderRequest
↓
DatasetRequest
```

Anonymous payloads are prohibited.

---

# 157. Response Models

Every endpoint returns structured responses.

Standard format

```
status

data

metadata

errors
```

Example

```
{
  "status":"success",
  "data":{},
  "metadata":{},
  "errors":[]
}
```

---

# 158. Error Responses

Every error follows the same schema.

```
code

message

hint

request_id
```

Example

```
{
  "code":"DATASET_NOT_FOUND",
  "message":"Requested dataset could not be located.",
  "hint":"Verify dataset path.",
  "request_id":"abc123"
}
```

---

# 159. State Philosophy

Frontend state and backend state are independent.

Frontend owns UI.

Backend owns business logic.

The backend never stores UI state.

---

# 160. Frontend State Categories

Three categories.

```
UI State
↓
Application State
↓
Server State
```

---

# 161. UI State

Examples

- dialog open
- theme
- zoom level
- sidebar collapsed

Managed locally.

---

# 162. Application State

Examples

- selected provider
- selected event
- selected frames
- interpolation settings

Managed by Zustand.

---

# 163. Server State

Examples

- datasets
- provider list
- health
- metrics
- generated result

Managed by TanStack Query.

Server state is never duplicated.

---

# 164. Communication Services

The frontend communicates only through services.

```
SystemService

DatasetService

InterpolationService

MetricsService

ExportService
```

Components never call fetch() directly.

---

# 165. Hook Layer

Hooks connect services to UI.

Example

```
Component
↓
Hook
↓
Service
↓
Backend
```

Hooks contain interaction logic.

---

# 166. Event Flow

The application is event-driven.

Example

```
Upload Completed
↓
Validation Started
↓
Interpolation Started
↓
Interpolation Completed
↓
Viewer Updated
```

Events describe state transitions.

---

# 167. Progress Events

Long-running tasks expose progress.

Example

```
Uploading
↓
Loading Dataset
↓
Preparing Tensors
↓
Running Model
↓
Saving Output
↓
Completed
```

These events improve user feedback.

---

# 168. Request IDs

Every request receives a unique Request ID.

Example

```
FRM-20260705-000145
```

The ID propagates through:

- logs
- metrics
- responses
- exports

This enables end-to-end traceability.

---

# 169. Timeouts

Version 1 default timeout

```
60 seconds
```

If exceeded

Return

```
408 Request Timeout
```

Long-running operations should eventually migrate to asynchronous jobs.

---

# 170. Asynchronous Operations

Future asynchronous tasks include:

- GIF generation
- PDF report generation
- dataset indexing
- cache cleanup
- batch interpolation

Version 1 interpolation remains synchronous.

---

# 171. Caching Strategy

Cache only:

- provider metadata
- dataset indices
- system information

Never cache:

- uploaded images
- interpolation requests
- generated tensors

unless explicitly configured.

---

# 172. Communication Security

All communication shall:

- validate input
- sanitize filenames
- reject oversized payloads
- reject unsupported formats
- avoid exposing internal paths

The frontend is never trusted.

---

# 173. API Documentation

Every endpoint must include:

- description
- request schema
- response schema
- error codes
- examples

FastAPI automatically exposes OpenAPI documentation.

---

# 174. Observability

Every request records:

- request ID
- timestamp
- provider
- model
- runtime
- status
- endpoint

This information supports debugging and monitoring.

---

# 175. Future Communication

The architecture allows future support for:

- WebSockets
- Server-Sent Events
- Background workers
- Message queues

without changing application services.

---

# 176. Communication Summary

The Communication Architecture guarantees that every subsystem exchanges information through explicit, typed, versioned contracts.

No subsystem depends on another subsystem's internal implementation.

This architecture ensures:

- maintainability
- extensibility
- observability
- scalability

while preserving a clean separation between presentation, orchestration, AI processing, and storage.

---

# End of Part 7

# ADR-001 — System Architecture

# Part 8 — Cross-Cutting Concerns

---

# 177. Purpose

Cross-cutting concerns are capabilities that affect every subsystem in FrameSat AI.

Unlike business logic, these concerns are shared across:

- Frontend
- Backend
- AI Pipeline
- Providers
- Storage

This ADR standardizes how these concerns are implemented.

---

# 178. Cross-Cutting Areas

FrameSat AI defines eight cross-cutting domains.

```

Configuration

↓

Dependency Injection

↓

Logging

↓

Error Handling

↓

Security

↓

Performance

↓

Monitoring

↓

Resource Management

```

Every subsystem shall follow these standards.

---

# 179. Configuration Philosophy

No configurable value shall be hardcoded.

Configuration must be externalized.

Sources of configuration:

```
Environment Variables

↓

Configuration Files

↓

Runtime Overrides
```

Priority is applied in that order.

---

# 180. Configuration Categories

Configuration is grouped into domains.

Examples

```
Application

Model

Provider

Storage

Logging

Security

Deployment
```

Each domain owns its own configuration object.

---

# 181. Environment Variables

Environment variables are used for deployment-specific values.

Examples

```
DEVICE

MODEL_PATH

OUTPUT_PATH

CACHE_PATH

LOG_LEVEL

MAX_UPLOAD_SIZE

API_PORT
```

Environment variables are validated during startup.

---

# 182. Dependency Injection

FrameSat AI uses Dependency Injection to reduce coupling.

Every service receives its dependencies.

Never instantiate dependencies inside business logic.

Correct

```
InterpolationService

↓

DataProvider

↓

Model
```

Incorrect

```
InterpolationService

↓

new SEVIRProvider()
```

---

# 183. Dependency Container

A central dependency container is responsible for:

- configuration
- providers
- services
- models

This container is initialized once during application startup.

---

# 184. Logging Philosophy

Logging is a first-class architectural feature.

Logs are intended for:

- debugging
- monitoring
- auditing
- performance analysis

Logging is never optional.

---

# 185. Log Levels

The application supports five log levels.

```
DEBUG

INFO

WARNING

ERROR

CRITICAL
```

Production defaults to INFO.

Development defaults to DEBUG.

---

# 186. Structured Logging

All backend logs follow a structured format.

Required fields

```
timestamp

request_id

module

level

message
```

Optional fields

```
provider

dataset

runtime

model

device
```

Logs must be machine-readable.

---

# 187. Request Tracing

Every request receives a globally unique Request ID.

The Request ID is propagated through:

- API
- Services
- Providers
- AI Pipeline
- Metrics
- Logs

This enables complete request tracing.

---

# 188. Error Handling Philosophy

Errors are expected.

Failures must never terminate the application.

Instead:

```
Detect

↓

Classify

↓

Log

↓

Recover

↓

Report
```

The user should always receive a meaningful response.

---

# 189. Error Categories

Errors are categorized.

Examples

```
ValidationError

ProviderError

InferenceError

ConfigurationError

StorageError

SystemError
```

Each category has a standard response format.

---

# 190. Exception Propagation

Exceptions move upward through the architecture.

```
Pipeline

↓

Service

↓

API

↓

Frontend
```

Each layer enriches the error.

Internal stack traces remain internal.

---

# 191. User Error Messages

User-facing messages must:

- explain what happened
- explain why
- suggest a solution

Avoid technical jargon.

---

# 192. Security Philosophy

Security is implemented through layered defense.

Version 1 focuses on:

- input validation
- file safety
- configuration security
- resource isolation

Authentication is reserved for a future version.

---

# 193. Input Validation

Every external input is validated.

Examples

- uploaded files
- dataset names
- frame indices
- API payloads

Validation occurs before processing.

---

# 194. File Handling

Uploaded files must:

- use temporary directories
- receive generated filenames
- avoid directory traversal
- be deleted after processing

Never trust client-provided filenames.

---

# 195. Path Safety

All filesystem operations use canonical paths.

Relative path traversal is prohibited.

The backend must reject invalid paths.

---

# 196. Performance Philosophy

Performance improvements must never reduce correctness.

Optimization order:

```
Correctness

↓

Reliability

↓

Maintainability

↓

Performance
```

Premature optimization is avoided.

---

# 197. Performance Metrics

The backend records:

- request latency
- inference time
- preprocessing time
- postprocessing time
- memory usage
- CPU usage
- GPU usage (if available)

Metrics are stored separately from logs.

---

# 198. Memory Management

Large datasets shall be processed incrementally.

Avoid:

- loading entire datasets
- unnecessary image copies
- duplicate tensors

Memory ownership must be explicit.

---

# 199. Resource Lifecycle

Resources follow a defined lifecycle.

```
Acquire

↓

Use

↓

Release
```

Resources include:

- files
- dataset handles
- GPU memory
- temporary directories

Every resource owner is responsible for cleanup.

---

# 200. Monitoring

The system continuously exposes health information.

Health checks include:

- model availability
- provider availability
- storage accessibility
- GPU availability
- configuration validity

These checks are lightweight.

---

# 201. Startup Validation

Application startup performs validation.

Checks include:

- configuration
- model weights
- dataset availability
- output directories
- write permissions

Startup fails if critical dependencies are unavailable.

---

# 202. Shutdown Procedure

Graceful shutdown releases:

- providers
- model resources
- open files
- caches
- temporary data

No resources should remain allocated.

---

# 203. Feature Flags

Future features may be controlled using feature flags.

Examples

```
ENABLE_SEVIR

ENABLE_GPU

ENABLE_REPORTS

ENABLE_GIF_EXPORT
```

Feature flags allow controlled rollout without code changes.

---

# 204. Operational Modes

The application supports multiple modes.

```
Development

Testing

Production
```

Each mode defines:

- logging level
- cache behavior
- debug output
- error verbosity

---

# 205. Documentation Requirements

Every module must document:

- purpose
- inputs
- outputs
- dependencies
- exceptions
- examples

Documentation is part of the implementation.

---

# 206. Definition of Operational Readiness

A subsystem is operationally ready when:

- configuration validated
- logging enabled
- errors handled
- resources managed
- tests passing
- documentation complete

Operational readiness is required before release.

---

# 207. Cross-Cutting Summary

The cross-cutting architecture ensures consistent behavior across all layers of FrameSat AI.

By standardizing configuration, dependency management, logging, security, error handling, performance, monitoring, and resource management, the platform remains reliable, maintainable, and production-ready regardless of future feature growth.

---

# End of Part 8

# ADR-001 — System Architecture

# Part 9 — Scalability, Deployment Strategy, Architectural Trade-offs & Future Evolution

---

# 208. Purpose

This section defines how FrameSat AI should evolve after Version 1.0.

Rather than optimizing only for the Bharatiya Antariksh Hackathon, the architecture is designed to support future research, larger datasets, additional AI models, and production deployment.

Every architectural decision must balance:

- simplicity
- maintainability
- extensibility
- performance

---

# 209. Architectural Philosophy

FrameSat AI follows one core philosophy:

> Build for tomorrow while remaining simple today.

The architecture intentionally avoids unnecessary complexity in Version 1 while ensuring that future expansion does not require rewriting the system.

---

# 210. Scalability Principles

The system must scale in four dimensions.

## Functional Scalability

Adding new capabilities.

Examples

- New interpolation models
- Report generation
- Dataset comparison
- Multi-frame interpolation

---

## Data Scalability

Supporting larger datasets.

Examples

- SEVIR
- NOAA GOES
- INSAT
- Sentinel
- Himawari

The architecture must never assume one specific dataset.

---

## User Scalability

Future versions should support:

- multiple analysts
- authentication
- project workspaces
- saved sessions

These features are intentionally excluded from Version 1.

---

## Infrastructure Scalability

The system should eventually support:

- cloud deployment
- GPU servers
- container orchestration
- distributed processing

The current architecture keeps this path open.

---

# 211. Growth Strategy

FrameSat AI will evolve in stages.

```
Version 1

↓

Version 1.5

↓

Version 2

↓

Version 3
```

Every version adds capabilities without redesigning the architecture.

---

# 212. Version 1

Focus

Core interpolation platform.

Features

- Upload images
- SEVIR integration
- RIFE interpolation
- Viewer
- Export
- Metrics
- REST API

No authentication.

No cloud processing.

---

# 213. Version 1.5

Focus

Analyst productivity.

Possible additions

- interpolation history
- project management
- GIF generation
- report builder
- advanced metrics
- caching improvements

---

# 214. Version 2

Focus

Multiple AI models.

Examples

```
RIFE

FILM

AMT

IFRNet
```

Users can select the active interpolation model.

Model benchmarking becomes possible.

---

# 215. Version 3

Focus

Operational intelligence.

Potential capabilities

- temporal forecasting
- object detection
- wildfire tracking
- flood monitoring
- storm evolution
- damage assessment

These remain independent services built on top of the interpolation platform.

---

# 216. Deployment Philosophy

Version 1 deployment prioritizes simplicity.

Frontend

↓

Vercel

Backend

↓

Render

Model

↓

Local weights

Datasets

↓

Local storage

This minimizes operational overhead.

---

# 217. Future Deployment

As usage grows

the architecture supports

```
CDN

↓

Load Balancer

↓

Frontend

↓

API Gateway

↓

Backend Cluster

↓

Inference Workers

↓

Shared Storage

↓

Dataset Storage
```

No architectural redesign required.

---

# 218. Containerization

Every subsystem should eventually run inside its own container.

Examples

```
Frontend

Backend

Inference

Dataset Services

Monitoring
```

Containers improve portability and reproducibility.

---

# 219. CI/CD Strategy

Every commit should automatically trigger:

- linting
- formatting
- unit tests
- integration tests
- build verification

Future deployments may automatically publish successful builds.

---

# 220. Monitoring Roadmap

Future operational monitoring includes:

- uptime
- request latency
- GPU utilization
- memory usage
- provider health
- model health
- API availability

Monitoring should be passive and lightweight.

---

# 221. Backup Strategy

Critical assets include:

- datasets
- configuration
- generated reports
- logs

Version 1 relies on local storage.

Future versions may introduce cloud backups.

---

# 222. Architectural Trade-offs

The following trade-offs were made intentionally.

## Layered Monolith

Chosen instead of microservices.

Reason

- simpler
- easier debugging
- faster development

---

## REST

Chosen instead of GraphQL.

Reason

- simplicity
- automatic documentation
- broad tooling support

---

## Provider Pattern

Chosen instead of dataset-specific logic.

Reason

- extensibility
- maintainability

---

## Model Registry

Chosen instead of direct model creation.

Reason

- future model replacement

---

## Monorepo

Chosen instead of multiple repositories.

Reason

- unified versioning
- simpler onboarding
- synchronized releases

---

# 223. Known Limitations

Version 1 intentionally accepts the following limitations.

- synchronous inference
- single-user workflow
- local datasets
- local model weights
- no authentication
- no distributed execution

These limitations simplify the initial implementation.

---

# 224. Risks

## Large Datasets

Risk

Memory exhaustion.

Mitigation

Lazy loading.

---

## GPU Availability

Risk

No CUDA device.

Mitigation

Automatic CPU fallback.

---

## Long Inference Times

Risk

Poor user experience.

Mitigation

Progress indicators and optimized preprocessing.

---

## Corrupted Data

Risk

Pipeline failures.

Mitigation

Provider validation and input verification.

---

## Model Changes

Risk

Breaking existing workflows.

Mitigation

Model abstraction through BaseInterpolator.

---

# 225. Technical Debt Policy

Temporary solutions are allowed only if:

- documented
- isolated
- scheduled for replacement

Undocumented technical debt is prohibited.

---

# 226. Extensibility Rules

Adding a feature should require extending the system rather than modifying existing modules.

Examples

New dataset

↓

New Provider

New model

↓

New Wrapper

New metric

↓

New Metric Class

No existing implementations should require changes.

---

# 227. Architectural Governance

Future architectural changes require:

- new ADR
- documented rationale
- impact analysis
- migration plan

Major architectural decisions are never made informally.

---

# 228. Release Strategy

Every release follows:

```
Development

↓

Testing

↓

Release Candidate

↓

Production
```

Version numbers follow Semantic Versioning.

Example

```
1.0.0

1.1.0

1.2.0

2.0.0
```

---

# 229. Success Criteria

FrameSat AI Version 1 is considered successful when:

- both Upload and SEVIR workflows operate correctly
- interpolation is reliable
- frontend and backend are deployed
- documentation is complete
- automated tests pass
- the system can be demonstrated without manual fixes

---

# 230. Long-Term Vision

FrameSat AI should evolve from an interpolation platform into a comprehensive Earth Observation AI Workspace.

Future modules may include:

- disaster monitoring
- temporal forecasting
- anomaly detection
- geospatial analytics
- AI-assisted reporting

The current architecture intentionally supports this evolution without requiring foundational redesign.

---

# End of Part 9

---

# ADR-001 — System Architecture

# Part 10 — Architecture Governance, Implementation Rules & Definition of Done

---

# 231. Purpose

This section formally concludes ADR-001.

It establishes the governance model, implementation rules, architecture acceptance criteria, glossary, and project-wide Definition of Done.

Upon approval, ADR-001 becomes the primary architectural reference for FrameSat AI Version 1.0.

All future architectural decisions shall reference this document.

---

# 232. Architecture Governance

Architecture is governed through Architecture Decision Records (ADRs).

No major architectural change may occur without:

- documenting the proposed change
- explaining the rationale
- evaluating impacts
- defining a migration strategy

Examples requiring a new ADR include:

- replacing FastAPI
- adopting GraphQL
- changing the provider architecture
- replacing the AI pipeline
- moving to microservices

Minor implementation changes do not require new ADRs.

---

# 233. Decision Hierarchy

When conflicts occur, decisions are resolved in the following order:

1. Product Vision
2. ADRs
3. SRS
4. PRDs
5. Sprint Tasks
6. Source Code

If source code contradicts an ADR, the ADR is considered authoritative until formally revised.

---

# 234. Architecture Review Process

Every new subsystem must answer the following questions before implementation:

- Does it align with the product vision?
- Does it violate any dependency rule?
- Does it duplicate existing functionality?
- Can it be extended in the future?
- Can it be tested independently?
- Does it increase unnecessary complexity?

If the answer to any question is unsatisfactory, the design should be reconsidered before implementation.

---

# 235. Coding Standards

Backend

- Python 3.11+
- Type hints required
- Pydantic models for API contracts
- Docstrings for public interfaces
- Black formatting
- Ruff linting
- Pytest for testing

Frontend

- TypeScript only
- Functional React components
- ESLint
- Prettier
- Strict type checking
- No use of `any` unless justified

General

- Small, focused modules
- Meaningful naming
- Clear separation of concerns
- No duplicated logic

---

# 236. Git Workflow

The repository follows Git Flow.

```
main

↓

develop

↓

feature/<name>

↓

Pull Request

↓

Review

↓

Merge
```

Every feature branch should address one logical change.

Commits should be atomic and descriptive.

Examples

```
feat(provider): implement SEVIRProvider

fix(api): validate upload dimensions

docs(adr): add provider architecture
```

---

# 237. Documentation Policy

Documentation is developed alongside code.

Every public module must include:

- purpose
- responsibilities
- inputs
- outputs
- dependencies
- exceptions
- usage example

Project documentation is treated as a deliverable, not an afterthought.

---

# 238. Testing Policy

Every feature requires appropriate testing.

Minimum expectations:

- Unit Tests
- Integration Tests
- Manual Verification

Future additions may include:

- End-to-End Tests
- Performance Benchmarks
- Regression Tests

A feature is not complete until it is tested.

---

# 239. Performance Guidelines

Target objectives for Version 1:

Backend

- API response (excluding inference): < 200 ms

Inference

- Dependent on hardware
- Runtime always reported to the user

Frontend

- Initial page load < 3 seconds
- Responsive interactions
- Smooth animations at 60 FPS where practical

These values are goals rather than strict guarantees.

---

# 240. Security Guidelines

The application must:

- validate all external input
- sanitize uploaded filenames
- isolate temporary files
- avoid exposing filesystem paths
- avoid exposing stack traces
- never trust client-provided metadata

Secrets must never be committed to the repository.

---

# 241. Definition of Done (Module)

A module is considered complete only when:

- Functional requirements implemented
- Type checking passes
- Linting passes
- Unit tests pass
- Integrated successfully
- Documentation complete
- Logging implemented
- Error handling implemented
- Code reviewed
- No known critical defects

---

# 242. Definition of Done (Sprint)

A sprint is complete when:

- All planned tasks are finished
- Regression testing succeeds
- Documentation updated
- No blocking issues remain
- Demo can be performed successfully

---

# 243. Definition of Done (Project)

FrameSat AI Version 1.0 is complete when all of the following are true:

Core Features

- Upload workflow operational
- SEVIR workflow operational
- AI interpolation operational
- Metadata extraction operational
- Metrics displayed
- Export functions operational

User Experience

- Landing page complete
- Workspace complete
- Comparison viewer complete
- Timeline complete
- Responsive layout
- Accessible interface

Engineering

- Backend deployed
- Frontend deployed
- Logging enabled
- Health monitoring enabled
- Automated tests passing

Documentation

- README complete
- ADRs complete
- API documentation complete
- User guide complete
- Developer guide complete

Presentation

- Demonstration script prepared
- Architecture diagrams prepared
- Project presentation complete

---

# 244. Risks Accepted

Version 1 intentionally accepts:

- Single-user operation
- Local dataset storage
- Local model weights
- Synchronous inference
- No authentication
- Limited cloud scalability

These decisions reduce complexity while preserving future extensibility.

---

# 245. Architecture Success Criteria

The architecture is considered successful if:

- New providers can be added without modifying the AI pipeline.
- New AI models can be integrated without changing frontend code.
- The frontend remains independent of backend implementation details.
- Every subsystem is independently testable.
- No circular dependencies exist.
- Future extensions require extension rather than modification.

---

# 246. Future ADR Roadmap

Following ADR-001, the project will define:

- ADR-002 — Provider Architecture
- ADR-003 — AI Pipeline
- ADR-004 — Backend Architecture
- ADR-005 — Frontend Architecture
- ADR-006 — API Contracts
- ADR-007 — UI/UX Principles
- ADR-008 — Testing Strategy
- ADR-009 — Deployment Strategy
- ADR-010 — Engineering Standards

These documents provide detailed guidance while remaining consistent with ADR-001.

---

# 247. Glossary

**Provider**
A module responsible for retrieving data from a specific source.

**Frame**
A single satellite observation.

**FramePair**
Two consecutive observations used for interpolation.

**FrameSequence**
An ordered collection of temporal observations.

**Interpolation**
The generation of an intermediate observation between two existing observations.

**Pipeline**
The ordered sequence of validation, preprocessing, inference, postprocessing, and metrics.

**Workspace**
The primary user interface where analysts perform tasks.

**ADR**
Architecture Decision Record.

**PRD**
Product Requirements Document.

---

# 248. Architecture Approval

ADR-001 is approved upon acceptance by the project maintainers.

Subsequent implementation shall conform to the principles and constraints documented herein.

Any deviation requires a new Architecture Decision Record.

---

# 249. Final Statement

FrameSat AI is designed as a modular, extensible, and production-oriented satellite temporal intelligence platform.

The architecture intentionally separates concerns between presentation, orchestration, AI processing, data providers, and storage.

By emphasizing clear interfaces, replaceable components, and rigorous engineering practices, the system provides a stable foundation for both Version 1.0 and future evolution into a broader Earth Observation AI platform.

This document serves as the constitutional reference for the FrameSat AI codebase.

---

**Document Status:** Accepted

**Version:** 1.0.0

**Last Updated:** July 2026

**End of ADR-001 — System Architecture**