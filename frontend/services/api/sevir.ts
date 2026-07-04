import { API_BASE_URL, apiRequest } from "@/services/api/client";
import type {
  InterpolateResponse,
  SevirEvent,
  SevirEventFrames,
  SevirEventsPage,
} from "@/types";

interface RawSevirEvent {
  event_id: string;
  name: string;
  date: string;
  time_utc: string;
  img_types: string[];
  episode_id: number | null;
  nws_event_id: number | null;
  bbox: {
    llcrnrlat: number;
    llcrnrlon: number;
    urcrnrlat: number;
    urcrnrlon: number;
  } | null;
}

interface RawSevirEventsPage {
  total: number;
  page: number;
  per_page: number;
  events: RawSevirEvent[];
}

interface RawSevirEventFrames {
  event_id: string;
  img_type: string;
  frames: { index: number; offset_minutes: number; timestamp: string }[];
}

interface RawSevirInterpolateResponse {
  image_url: string;
  runtime: number;
  resolution: string;
  device: string;
  model: string;
  event_id: string;
  img_type: string;
  frame_a: number;
  frame_b: number;
  ground_truth_frame: number | null;
  psnr: number | null;
  ssim: number | null;
  lpips: number | null;
}

function mapEvent(raw: RawSevirEvent): SevirEvent {
  return {
    eventId: raw.event_id,
    name: raw.name,
    date: raw.date,
    timeUtc: raw.time_utc,
    imgTypes: raw.img_types,
    episodeId: raw.episode_id,
    nwsEventId: raw.nws_event_id,
    bbox: raw.bbox,
  };
}

/** GET /datasets/sevir/events -- browse/search SEVIR storm events. */
export async function fetchSevirEvents(params?: {
  year?: number;
  imgType?: string;
  page?: number;
  perPage?: number;
}): Promise<SevirEventsPage> {
  const query = new URLSearchParams();
  if (params?.year) query.set("year", String(params.year));
  if (params?.imgType) query.set("img_type", params.imgType);
  if (params?.page) query.set("page", String(params.page));
  if (params?.perPage) query.set("per_page", String(params.perPage));

  const qs = query.toString();
  const raw = await apiRequest<RawSevirEventsPage>(
    `/datasets/sevir/events${qs ? `?${qs}` : ""}`,
  );

  return {
    total: raw.total,
    page: raw.page,
    perPage: raw.per_page,
    events: raw.events.map(mapEvent),
  };
}

/** GET /datasets/sevir/events/{id}/frames -- frame timeline for one event. */
export async function fetchSevirEventFrames(
  eventId: string,
  imgType: string,
): Promise<SevirEventFrames> {
  const raw = await apiRequest<RawSevirEventFrames>(
    `/datasets/sevir/events/${encodeURIComponent(eventId)}/frames?img_type=${encodeURIComponent(imgType)}`,
  );

  return {
    eventId: raw.event_id,
    imgType: raw.img_type,
    frames: raw.frames.map((f) => ({
      index: f.index,
      offsetMinutes: f.offset_minutes,
      timestamp: f.timestamp,
    })),
  };
}

/** Builds the absolute preview-thumbnail URL for a given SEVIR frame. */
export async function fetchSevirFramePreviewUrl(
  eventId: string,
  imgType: string,
  frameIndex: number,
): Promise<string> {
  const raw = await apiRequest<{ preview_url: string }>(
    `/datasets/sevir/events/${encodeURIComponent(eventId)}/frames/${frameIndex}/preview?img_type=${encodeURIComponent(imgType)}`,
  );
  return raw.preview_url.startsWith("http")
    ? raw.preview_url
    : `${API_BASE_URL}${raw.preview_url}`;
}

/** POST /interpolate/sevir -- run interpolation on two SEVIR-provided frames. */
export async function interpolateSevirFrames(params: {
  eventId: string;
  imgType: string;
  frameA: number;
  frameB: number;
}): Promise<InterpolateResponse> {
  const raw = await apiRequest<RawSevirInterpolateResponse>("/interpolate/sevir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      event_id: params.eventId,
      img_type: params.imgType,
      frame_a: params.frameA,
      frame_b: params.frameB,
    }),
  });

  const imageUrl = raw.image_url.startsWith("http")
    ? raw.image_url
    : `${API_BASE_URL}${raw.image_url}`;

  return {
    imageUrl,
    runtime: raw.runtime,
    resolution: raw.resolution,
    device: raw.device,
    model: raw.model,
    eventId: raw.event_id,
    imgType: raw.img_type,
    frameA: raw.frame_a,
    frameB: raw.frame_b,
    groundTruthFrame: raw.ground_truth_frame,
    psnr: raw.psnr,
    ssim: raw.ssim,
    lpips: raw.lpips,
  };
}
