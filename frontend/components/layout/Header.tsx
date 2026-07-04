"use client";

import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { SatelliteDish, Menu } from "lucide-react";
import { fetchSystemInfo } from "@/services/api/system";
import { useSettingsStore } from "@/store/useSettingsStore";

export function Header() {
  const toggleSidebar = useSettingsStore((s) => s.toggleSidebar);
  const datasetId = useSettingsStore((s) => s.datasetId);
  const setDataset = useSettingsStore((s) => s.setDataset);

  const { data } = useQuery({
    queryKey: ["system-info"],
    queryFn: fetchSystemInfo,
    staleTime: 30_000,
    retry: 1,
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b-4 border-paper pb-6"
    >
      <div className="flex items-center gap-3">
        <button
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          className="border-2 border-paper p-2 text-paper hover:bg-paper hover:text-ink lg:hidden"
        >
          <Menu size={18} />
        </button>
        <div>
          <h1 className="font-display text-3xl font-black uppercase tracking-tighter sm:text-4xl">
            Temporal Engine
          </h1>
          <p className="mt-1 font-mono text-xs text-cyan">
            MODEL: {data?.model?.toUpperCase() ?? "PRACTICAL-RIFE"} · STATUS:{" "}
            {data?.status ?? "READY"}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 border-2 border-paper bg-ink px-3 py-2">
          <span className="font-display text-xs font-bold uppercase text-paper">
            Source:
          </span>
          <select
            aria-label="Dataset source"
            value={datasetId ?? "default"}
            onChange={(e) => setDataset(e.target.value)}
            className="cursor-pointer border-none bg-transparent font-mono text-xs uppercase text-green outline-none focus:ring-0"
          >
            <option value="default">Default Dataset</option>
          </select>
        </div>

        <div className="hidden items-center gap-2 border-2 border-paper px-3 py-2 font-mono text-xs uppercase text-paper sm:flex">
          {data?.backend ?? "PyTorch"} · {data?.device ?? "CPU"}
        </div>

        <button
          aria-label="Mission control (coming soon)"
          disabled
          className="border-2 border-paper p-2.5 text-paper opacity-50 hover:bg-paper hover:text-ink"
        >
          <SatelliteDish size={18} />
        </button>
      </div>
    </motion.div>
  );
}
