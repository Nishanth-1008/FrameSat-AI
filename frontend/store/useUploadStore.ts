import { create } from "zustand";
import type { UploadedFrame, FrameSlot } from "@/types";

interface UploadState {
  frameA: UploadedFrame | null;
  frameB: UploadedFrame | null;
  setFrame: (slot: FrameSlot, frame: UploadedFrame) => void;
  clearFrame: (slot: FrameSlot) => void;
  clearAll: () => void;
  isComplete: () => boolean;
}

export const useUploadStore = create<UploadState>((set, get) => ({
  frameA: null,
  frameB: null,

  setFrame: (slot, frame) =>
    set((state) => {
      const previous = state[slot];
      if (previous) URL.revokeObjectURL(previous.previewUrl);
      return { [slot]: frame } as Partial<UploadState>;
    }),

  clearFrame: (slot) =>
    set((state) => {
      const previous = state[slot];
      if (previous) URL.revokeObjectURL(previous.previewUrl);
      return { [slot]: null } as Partial<UploadState>;
    }),

  clearAll: () =>
    set((state) => {
      if (state.frameA) URL.revokeObjectURL(state.frameA.previewUrl);
      if (state.frameB) URL.revokeObjectURL(state.frameB.previewUrl);
      return { frameA: null, frameB: null };
    }),

  isComplete: () => Boolean(get().frameA && get().frameB),
}));
