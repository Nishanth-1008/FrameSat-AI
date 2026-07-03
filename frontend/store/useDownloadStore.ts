import { create } from "zustand";

interface DownloadState {
  isDownloading: boolean;
  lastDownloadedUrl: string | null;
  startDownload: () => void;
  finishDownload: (url: string) => void;
}

export const useDownloadStore = create<DownloadState>((set) => ({
  isDownloading: false,
  lastDownloadedUrl: null,
  startDownload: () => set({ isDownloading: true }),
  finishDownload: (url) =>
    set({ isDownloading: false, lastDownloadedUrl: url }),
}));
