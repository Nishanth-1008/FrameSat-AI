"use client";

import { motion } from "framer-motion";
import { FrameDropzone } from "@/components/upload/FrameDropzone";
import { GenerateButton } from "@/components/dashboard/GenerateButton";

export function UploadWorkspace() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="border-4 border-paper bg-paper p-6 text-ink shadow-brutalist-lg"
    >
      <div className="mb-6 flex items-center justify-between border-b-4 border-ink pb-4">
        <h2 className="font-display text-2xl font-black uppercase tracking-tighter">
          Interpolation Pipeline
        </h2>
        <div className="font-mono text-xs font-bold uppercase">
          Status: <span className="text-cyan">Ready</span>
        </div>
      </div>

      <div className="grid grid-cols-1 items-center gap-6 lg:grid-cols-[1fr_auto_1fr]">
        <FrameDropzone slot="frameA" label="Frame A" />
        <div className="flex justify-center">
          <GenerateButton />
        </div>
        <FrameDropzone slot="frameB" label="Frame B" />
      </div>
    </motion.section>
  );
}
