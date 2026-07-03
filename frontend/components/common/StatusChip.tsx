import clsx from "clsx";
import type { BackendStatus } from "@/types";

const STATUS_STYLES: Record<BackendStatus, string> = {
  READY: "text-green border-green",
  BUSY: "text-cyan border-cyan",
  OFFLINE: "text-muted border-muted",
  ERROR: "text-alert border-alert",
};

export function StatusChip({
  status,
  label,
}: {
  status: BackendStatus;
  label?: string;
}) {
  return (
    <div
      className={clsx(
        "inline-flex items-center gap-2 border-2 bg-ink px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-widest",
        STATUS_STYLES[status],
      )}
      role="status"
    >
      <span
        className={clsx(
          "h-2 w-2 rounded-full",
          status === "READY" && "bg-green animate-pulseDot",
          status === "BUSY" && "bg-cyan animate-pulseDot",
          status === "OFFLINE" && "bg-muted",
          status === "ERROR" && "bg-alert",
        )}
      />
      {label ?? status}
    </div>
  );
}
