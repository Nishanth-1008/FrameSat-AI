"use client";

import clsx from "clsx";
import { Upload, Satellite } from "lucide-react";
import { useDataSourceStore } from "@/store/useDataSourceStore";

export function DataSourceTabs() {
  const { mode, setMode } = useDataSourceStore();

  const tabs = [
    { id: "upload" as const, label: "Upload Images", icon: Upload },
    { id: "sevir" as const, label: "Browse SEVIR", icon: Satellite },
  ];

  return (
    <div className="mb-6 flex gap-3">
      {tabs.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => setMode(id)}
          className={clsx(
            "flex flex-1 items-center justify-center gap-2 border-4 py-3 font-display text-sm font-black uppercase tracking-tight transition-colors",
            mode === id
              ? "border-paper bg-paper text-ink shadow-brutalist"
              : "border-paper/40 bg-panel text-paper/60 hover:border-paper hover:text-paper",
          )}
        >
          <Icon size={18} />
          {label}
        </button>
      ))}
    </div>
  );
}
