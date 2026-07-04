export type BackendStatus = "READY" | "BUSY" | "OFFLINE" | "ERROR";

export interface SystemInfo {
  model: string;
  backend: string;
  device: "CUDA" | "CPU" | string;
  version: string;
  status: BackendStatus;
}

export interface QualityMetrics {
  /** Peak signal-to-noise ratio in dB. Higher is better. Only present when ground truth exists (SEVIR mode). */
  psnr: number | null;
  /** Structural similarity, 0-1. Higher is better. */
  ssim: number | null;
  /** Learned Perceptual Image Patch Similarity. Lower is better. May be null if unavailable server-side. */
  lpips: number | null;
}

export interface InterpolateResponse extends Partial<QualityMetrics> {
  /** URL (relative or absolute) to the generated frame image */
  imageUrl: string;
  runtime: number; // seconds
  resolution: string; // e.g. "512 x 512"
  device: string;
  model: string;
  /** Present only for SEVIR-mode results */
  eventId?: string;
  imgType?: string;
  frameA?: number;
  frameB?: number;
  groundTruthFrame?: number | null;
}

export interface UploadedFrame {
  file: File;
  previewUrl: string;
}

export type FrameSlot = "frameA" | "frameB";

export type DataSourceMode = "upload" | "sevir";

export interface ApiErrorShape {
  message: string;
  status?: number;
}

// -- SEVIR dataset types ----------------------------------------------------

export const SEVIR_IMG_TYPES = ["vil", "vis", "ir069", "ir107"] as const;
export type SevirImgType = (typeof SEVIR_IMG_TYPES)[number];

export interface SevirEventBBox {
  llcrnrlat: number;
  llcrnrlon: number;
  urcrnrlat: number;
  urcrnrlon: number;
}

export interface SevirEvent {
  eventId: string;
  name: string;
  date: string; // YYYY-MM-DD
  timeUtc: string; // ISO timestamp
  imgTypes: string[];
  episodeId: number | null;
  nwsEventId: number | null;
  bbox: SevirEventBBox | null;
}

export interface SevirEventsPage {
  total: number;
  page: number;
  perPage: number;
  events: SevirEvent[];
}

export interface SevirFrame {
  index: number;
  offsetMinutes: number;
  timestamp: string; // ISO timestamp
}

export interface SevirEventFrames {
  eventId: string;
  imgType: string;
  frames: SevirFrame[];
}
