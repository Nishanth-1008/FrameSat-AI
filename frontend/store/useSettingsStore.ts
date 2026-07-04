import { create } from "zustand";

interface SettingsState {
  datasetId: string | null;
  sidebarOpen: boolean;
  setDataset: (id: string) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  datasetId: null,
  sidebarOpen: true,
  setDataset: (id) => set({ datasetId: id }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}));
