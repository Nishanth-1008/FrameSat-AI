"use client";

import { useMutation } from "@tanstack/react-query";
import { interpolateSevirFrames } from "@/services/api/sevir";
import { useSevirStore } from "@/store/useSevirStore";
import { useLoadingStore } from "@/store/useLoadingStore";
import { useResultStore } from "@/store/useResultStore";
import { useErrorStore } from "@/store/useErrorStore";
import { ApiError } from "@/services/api/client";

export function useSevirInterpolate() {
  const { selectedEvent, imgType, frameAIndex, frameBIndex } = useSevirStore();
  const { isProcessing, start, stop } = useLoadingStore();
  const { setResult } = useResultStore();
  const { pushError } = useErrorStore();

  const mutation = useMutation({
    mutationFn: async () => {
      if (!selectedEvent || frameAIndex === null || frameBIndex === null) {
        throw new Error("Select an event and both frames before generating.");
      }
      return interpolateSevirFrames({
        eventId: selectedEvent.eventId,
        imgType,
        frameA: frameAIndex,
        frameB: frameBIndex,
      });
    },
    onMutate: () => {
      start("Fetching SEVIR frames and running Practical-RIFE inference...");
    },
    onSuccess: (data) => {
      setResult(data);
    },
    onError: (error: unknown) => {
      const message =
        error instanceof ApiError
          ? error.message
          : error instanceof Error
            ? error.message
            : "Something went wrong while generating the frame.";

      pushError({
        message,
        kind: error instanceof ApiError ? "backend" : "validation",
      });
    },
    onSettled: () => {
      stop();
    },
  });

  const canGenerate =
    Boolean(selectedEvent) &&
    frameAIndex !== null &&
    frameBIndex !== null &&
    frameAIndex !== frameBIndex &&
    !isProcessing &&
    !mutation.isPending;

  return {
    generate: () => mutation.mutate(),
    isProcessing: isProcessing || mutation.isPending,
    canGenerate,
  };
}
