"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Gauge, Layers, Eye } from "lucide-react";
import { StatCard } from "@/components/cards/StatCard";
import { useResultStore } from "@/store/useResultStore";

export function QualityMetricsCards() {
  const { result } = useResultStore();

  const hasMetrics =
    result && (result.psnr !== undefined && result.psnr !== null);

  return (
    <AnimatePresence>
      {hasMetrics && (
        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="border-4 border-paper bg-panel p-6"
        >
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-display text-2xl font-black uppercase tracking-tighter text-paper">
              Ground-Truth Quality
            </h2>
            <span className="font-mono text-[10px] uppercase text-muted">
              vs. true SEVIR frame {result?.groundTruthFrame ?? ""}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StatCard
              icon={Gauge}
              label="PSNR (higher better)"
              value={result?.psnr != null ? `${result.psnr.toFixed(2)} dB` : "—"}
            />
            <StatCard
              icon={Layers}
              label="SSIM (higher better)"
              value={result?.ssim != null ? result.ssim.toFixed(3) : "—"}
            />
            <StatCard
              icon={Eye}
              label="LPIPS (lower better)"
              value={result?.lpips != null ? result.lpips.toFixed(3) : "N/A"}
            />
          </div>
        </motion.section>
      )}
    </AnimatePresence>
  );
}
