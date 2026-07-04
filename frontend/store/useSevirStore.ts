import { create } from "zustand";
import type { SevirEvent, SevirImgType } from "@/types";

interface SevirState {
  selectedEvent: SevirEvent | null;
  imgType: SevirImgType;
  frameAIndex: number | null;
  frameBIndex: number | null;

  selectEvent: (event: SevirEvent) => void;
  setImgType: (imgType: SevirImgType) => void;
  setFrameA: (index: number) => void;
  setFrameB: (index: number) => void;
  reset: () => void;
}

export const useSevirStore = create<SevirState>((set) => ({
  selectedEvent: null,
  imgType: "vil",
  frameAIndex: null,
  frameBIndex: null,

  selectEvent: (event) =>
    set({ selectedEvent: event, frameAIndex: null, frameBIndex: null }),
  setImgType: (imgType) =>
    set({ imgType, frameAIndex: null, frameBIndex: null }),
  setFrameA: (index) => set({ frameAIndex: index }),
  setFrameB: (index) => set({ frameBIndex: index }),
  reset: () =>
    set({ selectedEvent: null, frameAIndex: null, frameBIndex: null }),
}));
