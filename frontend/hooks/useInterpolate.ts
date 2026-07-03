"use client";

import { useMutation } from "@tanstack/react-query";
import { interpolateFrames } from "@/services/api/interpolate";
import { useUploadStore } from "@/store/useUploadStore";
import { useLoadingStore } from "@/store/useLoadingStore";
import { useResultStore } from "@/store/useResultStore";
import { useErrorStore } from "@/store/useErrorStore";
import { ApiError } from "@/services/api/client";

export function useInterpolate() {
  const { frameA, frameB } = useUploadStore();
  const { isProcessing, start, stop } = useLoadingStore();
  const { setResult } = useResultStore();
  const { pushError } = useErrorStore();

  const mutation = useMutation({
    mutationFn: async () => {
      if (!frameA || !frameB) {
        throw new Error("Both frames are required before generating.");
      }
      return interpolateFrames(frameA.file, frameB.file);
    },
    onMutate: () => {
      start("Running Practical-RIFE inference...");
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

  const canGenerate = Boolean(frameA && frameB) && !isProcessing && !mutation.isPending;

  return {
    generate: () => mutation.mutate(),
    isProcessing: isProcessing || mutation.isPending,
    canGenerate,
  };
}
