"use client";

import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, WifiOff, Clock, FileWarning, X } from "lucide-react";
import { useErrorStore, type AppError } from "@/store/useErrorStore";

const ICONS: Record<AppError["kind"], React.ElementType> = {
  network: WifiOff,
  validation: AlertTriangle,
  backend: AlertTriangle,
  timeout: Clock,
  unsupported: FileWarning,
};

export function ToastStack() {
  const { errors, dismissError } = useErrorStore();

  return (
    <div
      className="fixed bottom-6 right-6 z-50 flex w-full max-w-sm flex-col gap-3"
      aria-live="assertive"
    >
      <AnimatePresence>
        {errors.map((error) => {
          const Icon = ICONS[error.kind];
          return (
            <motion.div
              key={error.id}
              initial={{ opacity: 0, x: 40 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 40 }}
              role="alert"
              className="flex items-start gap-3 border-4 border-alert bg-ink p-4 shadow-brutalist-ink"
            >
              <Icon size={18} className="mt-0.5 shrink-0 text-alert" />
              <p className="flex-1 font-mono text-xs text-paper">{error.message}</p>
              <button
                onClick={() => dismissError(error.id)}
                aria-label="Dismiss notification"
                className="text-muted hover:text-paper"
              >
                <X size={16} />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
