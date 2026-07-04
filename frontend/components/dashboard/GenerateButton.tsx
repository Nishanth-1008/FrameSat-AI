"use client";

import { Loader2, Zap } from "lucide-react";
import { Button } from "@/components/common/Button";
import { useInterpolate } from "@/hooks/useInterpolate";
import { useLoadingStore } from "@/store/useLoadingStore";

export function GenerateButton() {
  const { generate, isProcessing, canGenerate } = useInterpolate();
  const progressLabel = useLoadingStore((s) => s.progressLabel);

  return (
    <div className="flex flex-col items-center gap-3 py-2">
      <Button
        variant="primary"
        size="lg"
        disabled={!canGenerate}
        onClick={generate}
        className="min-w-[200px]"
      >
        <span className="flex items-center justify-center gap-2">
          {isProcessing ? (
            <Loader2 className="animate-spin" size={20} />
          ) : (
            <Zap size={20} />
          )}
          {isProcessing ? "Generating" : "Generate"}
        </span>
      </Button>
      {isProcessing && (
        <p className="font-mono text-[11px] uppercase text-muted" aria-live="polite">
          {progressLabel}
        </p>
      )}
    </div>
  );
}
