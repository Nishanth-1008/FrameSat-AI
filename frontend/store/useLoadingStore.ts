import { create } from "zustand";

interface LoadingState {
  isProcessing: boolean;
  progressLabel: string;
  start: (label?: string) => void;
  setLabel: (label: string) => void;
  stop: () => void;
}

export const useLoadingStore = create<LoadingState>((set) => ({
  isProcessing: false,
  progressLabel: "",

  start: (label = "Running inference...") =>
    set({ isProcessing: true, progressLabel: label }),

  setLabel: (label) => set({ progressLabel: label }),

  stop: () => set({ isProcessing: false, progressLabel: "" }),
}));
