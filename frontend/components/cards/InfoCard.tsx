import { motion } from "framer-motion";
import clsx from "clsx";

export function InfoCard({
  label,
  value,
  loading = false,
}: {
  label: string;
  value: string;
  loading?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="border-2 border-ink bg-paper p-4 text-ink"
    >
      <span className="block font-mono text-[11px] uppercase tracking-widest text-ink/60">
        {label}
      </span>
      <span
        className={clsx(
          "mt-1.5 block font-display text-lg font-black uppercase",
          loading && "animate-pulse text-ink/40",
        )}
      >
        {loading ? "Loading…" : value}
      </span>
    </motion.div>
  );
}
