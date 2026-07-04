"use client";

import { motion } from "framer-motion";
import { Timer, Maximize, Cpu, BrainCircuit, Activity } from "lucide-react";
import { StatCard } from "@/components/cards/StatCard";
import { useResultStore } from "@/store/useResultStore";
import { useLoadingStore } from "@/store/useLoadingStore";

export function RuntimeCards() {
  const { result } = useResultStore();
  const { isProcessing } = useLoadingStore();

  const status = isProcessing ? "Processing" : result ? "Completed" : "Ready";

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="border-4 border-paper bg-panel p-6"
    >
      <h2 className="mb-4 font-display text-2xl font-black uppercase tracking-tighter text-paper">
        Runtime Telemetry
      </h2>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
        <StatCard icon={Activity} label="Status" value={status} />
        <StatCard
          icon={Timer}
          label="Inference Time"
          value={result ? `${result.runtime.toFixed(2)}s` : "—"}
        />
        <StatCard
          icon={Maximize}
          label="Resolution"
          value={result?.resolution ?? "—"}
        />
        <StatCard icon={Cpu} label="Device" value={result?.device ?? "—"} />
        <StatCard
          icon={BrainCircuit}
          label="Model"
          value={result?.model ?? "—"}
        />
      </div>
    </motion.section>
  );
}
