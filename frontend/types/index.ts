export type BackendStatus = "READY" | "BUSY" | "OFFLINE" | "ERROR";

export interface SystemInfo {
  model: string;
  backend: string;
  device: "CUDA" | "CPU" | string;
  version: string;
  status: BackendStatus;
}

export interface InterpolateResponse {
  /** URL (relative or absolute) to the generated frame image */
  imageUrl: string;
  runtime: number; // seconds
  resolution: string; // e.g. "512 x 512"
  device: string;
  model: string;
}

export interface UploadedFrame {
  file: File;
  previewUrl: string;
}

export type FrameSlot = "frameA" | "frameB";

export interface ApiErrorShape {
  message: string;
  status?: number;
}
