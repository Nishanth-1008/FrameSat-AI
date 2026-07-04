import { create } from "zustand";
import type { InterpolateResponse } from "@/types";

interface ResultState {
  result: InterpolateResponse | null;
  setResult: (result: InterpolateResponse) => void;
  clearResult: () => void;
}

export const useResultStore = create<ResultState>((set) => ({
  result: null,
  setResult: (result) => set({ result }),
  clearResult: () => set({ result: null }),
}));
