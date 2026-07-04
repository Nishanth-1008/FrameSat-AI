"use client";

import { useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { ImagePlus, X, RefreshCw } from "lucide-react";
import Image from "next/image";
import { useFrameUpload } from "@/hooks/useFrameUpload";
import type { FrameSlot } from "@/types";
import clsx from "clsx";

const TIMESTAMPS: Record<FrameSlot, string> = {
  frameA: "T-0",
  frameB: "T+30",
};

export function FrameDropzone({
  slot,
  label,
}: {
  slot: FrameSlot;
  label: string;
}) {
  const { frame, accept, remove } = useFrameUpload(slot);

  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];
      if (file) accept(file);
    },
    [accept],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    multiple: false,
    noClick: Boolean(frame),
    accept: {
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
      "image/webp": [".webp"],
      "image/tiff": [".tif", ".tiff"],
    },
  });

  useEffect(() => {
    function handlePaste(e: ClipboardEvent) {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/"),
      );
      const file = item?.getAsFile();
      if (file) accept(file);
    }
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [accept]);

  return (
    <div className="flex-1">
      <div className="mb-2 flex items-center justify-between border-2 border-ink bg-ink px-3 py-2 font-mono text-xs font-bold uppercase text-paper">
        <span>{label} ({TIMESTAMPS[slot]})</span>
        <span className="text-cyan">{frame ? "LOADED" : "PENDING"}</span>
      </div>
      <motion.div
        {...getRootProps()}
        className={clsx(
          "relative flex aspect-video w-full items-center justify-center overflow-hidden border-4 bg-panel transition-colors",
          isDragActive ? "border-cyan" : "border-ink",
          !frame && "cursor-pointer",
        )}
      >
        <input {...getInputProps()} aria-label={`Upload ${label}`} />

        {frame ? (
          <>
            <Image
              src={frame.previewUrl}
              alt={`${label} preview`}
              fill
              className="object-contain grayscale transition-all duration-300 hover:grayscale-0"
              unoptimized
            />
            <div className="absolute bottom-2 left-2 border border-cyan bg-ink px-1.5 py-0.5 font-mono text-[10px] text-cyan">
              RAW_IN
            </div>
            <div className="absolute right-2 top-2 flex gap-2">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  open();
                }}
                aria-label={`Replace ${label}`}
                className="border-2 border-paper bg-ink p-2 text-paper hover:bg-paper hover:text-ink"
              >
                <RefreshCw size={14} />
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  remove();
                }}
                aria-label={`Remove ${label}`}
                className="border-2 border-alert bg-ink p-2 text-alert hover:bg-alert hover:text-ink"
              >
                <X size={14} />
              </button>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center text-muted">
            <ImagePlus size={28} className="text-cyan" />
            <p className="font-mono text-xs uppercase">
              Drag, click, or paste image
            </p>
            <p className="text-[10px] text-muted/70">PNG · JPEG · WebP · TIFF, up to 25MB</p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
