# FrameSat AI — Frontend

Next.js 15 / React 19 / TypeScript dashboard that replaces the previous
Gradio UI, calling the existing (untouched) interpolation pipeline over
a thin REST layer.

## 1. Run the backend API

The original pipeline (`core/`, `app/services/`, `app/models/`) is
unchanged. A small FastAPI shim (`backend_api/api.py`, included
alongside this frontend) exposes it as REST:

```bash
pip install fastapi uvicorn python-multipart pillow
uvicorn backend_api.api:app --reload --port 8000
```

Endpoints:

- `GET /system` → `{ model, backend, device, version, status }`
- `POST /interpolate` (multipart: `frame_a`, `frame_b`) → `{ image_url, runtime, resolution, device, model }`
- `GET /outputs/<file>` → serves the generated frame

## 2. Run the frontend

```bash
cd frontend
cp .env.example .env.local   # set NEXT_PUBLIC_API_BASE_URL if not localhost:8000
npm install
npm run dev
```

Open http://localhost:3000.

## Structure

```
app/            routes, layout, providers, global styles
components/
  layout/       Sidebar, Header, Footer
  dashboard/    Dashboard, GenerateButton, RuntimeCards
  upload/       FrameDropzone, UploadWorkspace
  viewer/       ResultViewer (lazy-loaded)
  cards/        InfoCard, StatCard
  common/       Button, StatusChip, ToastStack
hooks/          useInterpolate, useFrameUpload
services/api/   client.ts, interpolate.ts, system.ts (only place fetch is called)
store/          Zustand: upload, loading, result, error, download, settings
types/          shared TypeScript types
```

No fake statistics, confidence values, or interpolation factors are
displayed — every number in the runtime cards comes from the live
`/interpolate` response.
