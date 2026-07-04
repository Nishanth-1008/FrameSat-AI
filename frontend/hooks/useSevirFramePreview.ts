"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchSevirFramePreviewUrl } from "@/services/api/sevir";

export function useSevirFramePreview(
  eventId: string | undefined,
  imgType: string,
  frameIndex: number | null,
) {
  const query = useQuery({
    queryKey: ["sevir-frame-preview", eventId, imgType, frameIndex],
    queryFn: () => fetchSevirFramePreviewUrl(eventId!, imgType, frameIndex!),
    enabled: Boolean(eventId) && frameIndex !== null,
    staleTime: 5 * 60_000,
  });

  return { previewUrl: query.data ?? null, isLoading: query.isLoading };
}
