"use client";

import { motion } from "framer-motion";
import { EventsBrowser } from "@/components/sevir/EventsBrowser";
import { FramePicker } from "@/components/sevir/FramePicker";
import { SevirGenerateButton } from "@/components/sevir/SevirGenerateButton";

export function SevirWorkspace() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="border-4 border-paper bg-paper p-6 text-ink shadow-brutalist-lg"
    >
      <div className="mb-6 flex items-center justify-between border-b-4 border-ink pb-4">
        <h2 className="font-display text-2xl font-black uppercase tracking-tighter">
          SEVIR Dataset Browser
        </h2>
        <div className="font-mono text-xs font-bold uppercase">
          Source: <span className="text-cyan">AWS Open Data</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <div className="h-[520px]">
          <EventsBrowser />
        </div>
        <div className="h-[520px]">
          <FramePicker />
        </div>
      </div>

      <div className="mt-4 flex justify-center">
        <SevirGenerateButton />
      </div>
    </motion.section>
  );
}
