"use client";

import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { Download, ImageOff } from "lucide-react";
import { useUploadStore } from "@/store/useUploadStore";
import { useResultStore } from "@/store/useResultStore";
import { useDownloadStore } from "@/store/useDownloadStore";
import { Button } from "@/components/common/Button";

function FramePanel({
  label,
  tag,
  src,
  empty,
  highlighted = false,
}: {
  label: string;
  tag: string;
  src: string | null;
  empty: string;
  highlighted?: boolean;
}) {
  return (
    <div
      className={
        highlighted
          ? "relative z-10 scale-[1.03] border-4 border-green bg-ink p-1 shadow-glowGreen"
          : "relative border-2 border-paper bg-ink p-1"
      }
    >
      {highlighted && <div className="scanline" />}
      <div className="relative flex aspect-video items-center justify-center overflow-hidden bg-panel">
        {src ? (
          <Image
            src={src}
            alt={label}
            fill
            className="object-contain grayscale transition-all duration-300 hover:grayscale-0"
            unoptimized
          />
        ) : (
          <div className="flex flex-col items-center gap-1 text-muted">
            <ImageOff size={22} />
            <span className="font-mono text-[10px] uppercase">{empty}</span>
          </div>
        )}
      </div>
      <div
        className={
          highlighted
            ? "absolute right-2 top-2 border-2 border-ink bg-green px-2 py-1 font-mono text-[10px] font-bold text-ink shadow-brutalist-ink"
            : "absolute right-2 top-2 border border-paper bg-ink px-1.5 py-0.5 font-mono text-[10px] text-paper"
        }
      >
        {tag}
      </div>
    </div>
  );
}

export function ResultViewer() {
  const { frameA, frameB } = useUploadStore();
  const { result } = useResultStore();
  const { startDownload, finishDownload } = useDownloadStore();

  async function handleDownload() {
    if (!result) return;
    startDownload();
    const response = await fetch(result.imageUrl);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "framesat_generated.png";
    link.click();
    URL.revokeObjectURL(url);
    finishDownload(result.imageUrl);
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.15 }}
      className="border-4 border-paper bg-panel p-6"
    >
      <div className="mb-6 flex items-center gap-4">
        <h2 className="font-display text-2xl font-black uppercase tracking-tighter text-paper">
          Resulting Sequence
        </h2>
        <span className="h-1 flex-1 bg-paper" />
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
            >
              <Button variant="secondary" onClick={handleDownload}>
                <span className="flex items-center gap-2">
                  <Download size={16} />
                  Download
                </span>
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <FramePanel
          label="Frame A"
          tag="T-0"
          src={frameA?.previewUrl ?? null}
          empty="No frame uploaded"
        />
        <FramePanel
          label="Generated Frame"
          tag="AI: T+15"
          src={result?.imageUrl ?? null}
          empty="Awaiting generation"
          highlighted
        />
        <FramePanel
          label="Frame B"
          tag="T+30"
          src={frameB?.previewUrl ?? null}
          empty="No frame uploaded"
        />
      </div>
    </motion.section>
  );
}
