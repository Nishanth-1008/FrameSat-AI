import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

export function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden border-4 border-paper bg-ink p-4"
    >
      <div className="flex items-center gap-2 text-muted">
        <Icon size={14} />
        <span className="font-mono text-[10px] uppercase tracking-widest">
          {label}
        </span>
      </div>
      <p className="mt-2 font-mono text-xl font-black text-cyan">{value}</p>
    </motion.div>
  );
}
