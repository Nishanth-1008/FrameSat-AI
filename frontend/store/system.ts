import { create } from "zustand";
import { getSystem, SystemInfo } from "@/services/api/system";

interface SystemState {
  info: SystemInfo | null;
  loading: boolean;
  error: string | null;

  load: () => Promise<void>;
}

export const useSystemStore = create<SystemState>((set) => ({
  info: null,
  loading: false,
  error: null,

  load: async () => {
    set({
      loading: true,
      error: null,
    });

    try {
      const data = await getSystem();

      set({
        info: data,
        loading: false,
      });
    } catch (err) {
      set({
        loading: false,
        error:
          err instanceof Error
            ? err.message
            : "Failed to connect to backend.",
      });
    }
  },
}));