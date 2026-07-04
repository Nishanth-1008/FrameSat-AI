"use client";

import Image from "next/image";
import { Clock, ImageOff } from "lucide-react";
import { useSevirStore } from "@/store/useSevirStore";
import { useSevirFrames } from "@/hooks/useSevirFrames";
import { useSevirFramePreview } from "@/hooks/useSevirFramePreview";
import { SEVIR_IMG_TYPES, type SevirImgType } from "@/types";

function FrameThumb({
  label,
  color,
  eventId,
  imgType,
  frameIndex,
  offsetMinutes,
}: {
  label: string;
  color: "cyan" | "green";
  eventId: string | undefined;
  imgType: string;
  frameIndex: number | null;
  offsetMinutes: number | undefined;
}) {
  const { previewUrl, isLoading } = useSevirFramePreview(eventId, imgType, frameIndex);
  const borderClass = color === "cyan" ? "border-cyan" : "border-green";

  return (
    <div className={`border-2 ${borderClass} bg-panel p-2`}>
      <div className="mb-1 flex items-center justify-between">
        <span className="font-mono text-[10px] font-bold uppercase text-paper">{label}</span>
        {offsetMinutes !== undefined && (
          <span className="flex items-center gap-1 font-mono text-[9px] text-muted">
            <Clock size={9} />
            {offsetMinutes >= 0 ? "+" : ""}
            {offsetMinutes}m
          </span>
        )}
      </div>
      <div className="relative flex aspect-square items-center justify-center overflow-hidden bg-ink">
        {frameIndex === null ? (
          <span className="font-mono text-[9px] text-muted">Not selected</span>
        ) : isLoading || !previewUrl ? (
          <span className="font-mono text-[9px] text-muted">Loading…</span>
        ) : (
          <Image src={previewUrl} alt={label} fill unoptimized className="object-contain" />
        )}
        {frameIndex === null && <ImageOff size={16} className="absolute text-muted" />}
      </div>
    </div>
  );
}

export function FramePicker() {
  const {
    selectedEvent,
    imgType,
    frameAIndex,
    frameBIndex,
    setImgType,
    setFrameA,
    setFrameB,
  } = useSevirStore();
  const { frames, isLoading } = useSevirFrames();

  if (!selectedEvent) {
    return (
      <div className="flex h-full items-center justify-center border-2 border-paper bg-ink p-6">
        <p className="font-mono text-xs text-muted">
          Select a storm event to choose frames.
        </p>
      </div>
    );
  }

  const maxIndex = Math.max(0, frames.length - 1);
  const frameAMeta = frameAIndex !== null ? frames[frameAIndex] : undefined;
  const frameBMeta = frameBIndex !== null ? frames[frameBIndex] : undefined;

  return (
    <div className="flex h-full flex-col gap-4 border-2 border-paper bg-ink p-4">
      <div className="flex items-center justify-between border-b-2 border-paper pb-3">
        <h3 className="font-display text-sm font-black uppercase tracking-tight text-paper">
          {selectedEvent.eventId}
        </h3>
        <select
          value={imgType}
          onChange={(e) => setImgType(e.target.value as SevirImgType)}
          className="border-2 border-paper bg-panel px-2 py-1 font-mono text-[10px] uppercase text-paper focus:border-cyan focus:outline-none"
        >
          {SEVIR_IMG_TYPES.filter((t) => selectedEvent.imgTypes.includes(t)).map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {isLoading ? (
        <p className="font-mono text-xs text-muted">Loading frame timeline…</p>
      ) : (
        <>
          <div>
            <label className="mb-1 flex justify-between font-mono text-[10px] uppercase text-cyan">
              <span>Frame A</span>
              <span>{frameAIndex ?? "—"} / {maxIndex}</span>
            </label>
            <input
              type="range"
              min={0}
              max={maxIndex}
              value={frameAIndex ?? 0}
              onChange={(e) => setFrameA(Number(e.target.value))}
              className="w-full accent-cyan"
            />
          </div>

          <div>
            <label className="mb-1 flex justify-between font-mono text-[10px] uppercase text-green">
              <span>Frame B</span>
              <span>{frameBIndex ?? "—"} / {maxIndex}</span>
            </label>
            <input
              type="range"
              min={0}
              max={maxIndex}
              value={frameBIndex ?? maxIndex}
              onChange={(e) => setFrameB(Number(e.target.value))}
              className="w-full accent-green"
            />
          </div>

          {frameAIndex !== null && frameBIndex !== null && frameAIndex === frameBIndex && (
            <p className="font-mono text-[10px] text-alert">
              Frame A and Frame B must be different.
            </p>
          )}

          <div className="grid grid-cols-2 gap-3">
            <FrameThumb
              label="Frame A"
              color="cyan"
              eventId={selectedEvent.eventId}
              imgType={imgType}
              frameIndex={frameAIndex}
              offsetMinutes={frameAMeta?.offsetMinutes}
            />
            <FrameThumb
              label="Frame B"
              color="green"
              eventId={selectedEvent.eventId}
              imgType={imgType}
              frameIndex={frameBIndex}
              offsetMinutes={frameBMeta?.offsetMinutes}
            />
          </div>

          {frameAIndex !== null &&
            frameBIndex !== null &&
            Math.abs(frameAIndex - frameBIndex) === 2 && (
              <p className="font-mono text-[10px] text-muted">
                Frames are 2 apart — the true middle frame will be used as
                ground truth for quality metrics (PSNR / SSIM / LPIPS).
              </p>
            )}
        </>
      )}
    </div>
  );
}
