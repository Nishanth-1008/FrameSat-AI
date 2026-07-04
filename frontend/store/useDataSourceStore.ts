import { create } from "zustand";
import type { DataSourceMode } from "@/types";

interface DataSourceState {
  mode: DataSourceMode;
  setMode: (mode: DataSourceMode) => void;
}

export const useDataSourceStore = create<DataSourceState>((set) => ({
  mode: "upload",
  setMode: (mode) => set({ mode }),
}));
