"use client";

import { useCallback } from "react";
import { useUploadStore } from "@/store/useUploadStore";
import { useErrorStore } from "@/store/useErrorStore";
import type { FrameSlot } from "@/types";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/tiff"];
const MAX_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB, generous for satellite frames

export function useFrameUpload(slot: FrameSlot) {
  const { setFrame, clearFrame, frameA, frameB } = useUploadStore();
  const { pushError } = useErrorStore();

  const frame = slot === "frameA" ? frameA : frameB;

  const validate = useCallback(
    (file: File): boolean => {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        pushError({
          kind: "unsupported",
          message: `"${file.name}" is not a supported image type (PNG, JPEG, WebP, TIFF).`,
        });
        return false;
      }
      if (file.size > MAX_SIZE_BYTES) {
        pushError({
          kind: "validation",
          message: `"${file.name}" exceeds the 25MB size limit.`,
        });
        return false;
      }
      return true;
    },
    [pushError],
  );

  const accept = useCallback(
    (file: File) => {
      if (!validate(file)) return;
      setFrame(slot, { file, previewUrl: URL.createObjectURL(file) });
    },
    [setFrame, slot, validate],
  );

  const remove = useCallback(() => clearFrame(slot), [clearFrame, slot]);

  return { frame, accept, remove };
}
