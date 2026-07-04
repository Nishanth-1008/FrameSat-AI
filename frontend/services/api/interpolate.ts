import { API_BASE_URL, apiRequest } from "@/services/api/client";
import type { InterpolateResponse } from "@/types";

interface RawInterpolateResponse {
  image_url: string;
  runtime: number;
  resolution: string;
  device: string;
  model: string;
}

/**
 * Calls POST /interpolate with the two source frames.
 * This is the single integration point with the interpolation pipeline.
 */
export async function interpolateFrames(
  frameA: File,
  frameB: File,
): Promise<InterpolateResponse> {
  const formData = new FormData();
  formData.append("frame_a", frameA);
  formData.append("frame_b", frameB);

  const raw = await apiRequest<RawInterpolateResponse>("/interpolate", {
    method: "POST",
    body: formData,
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
  };
}
