import { create } from "zustand";

export interface AppError {
  id: string;
  message: string;
  kind: "network" | "validation" | "backend" | "timeout" | "unsupported";
}

interface ErrorState {
  errors: AppError[];
  pushError: (error: Omit<AppError, "id">) => void;
  dismissError: (id: string) => void;
  clearErrors: () => void;
}

export const useErrorStore = create<ErrorState>((set) => ({
  errors: [],

  pushError: (error) =>
    set((state) => ({
      errors: [
        ...state.errors,
        { ...error, id: crypto.randomUUID() },
      ],
    })),

  dismissError: (id) =>
    set((state) => ({ errors: state.errors.filter((e) => e.id !== id) })),

  clearErrors: () => set({ errors: [] }),
}));
