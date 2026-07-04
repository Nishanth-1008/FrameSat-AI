"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchSevirEventFrames } from "@/services/api/sevir";
import { useSevirStore } from "@/store/useSevirStore";

export function useSevirFrames() {
  const { selectedEvent, imgType } = useSevirStore();

  const query = useQuery({
    queryKey: ["sevir-frames", selectedEvent?.eventId, imgType],
    queryFn: () => fetchSevirEventFrames(selectedEvent!.eventId, imgType),
    enabled: Boolean(selectedEvent),
    staleTime: 60_000,
  });

  return {
    frames: query.data?.frames ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
