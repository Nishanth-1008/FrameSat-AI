"use client";

import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { LayoutDashboard, Frame, Radio, Satellite, RotateCcw } from "lucide-react";
import { fetchSystemInfo } from "@/services/api/system";
import { StatusChip } from "@/components/common/StatusChip";
import { InfoCard } from "@/components/cards/InfoCard";
import { useSettingsStore } from "@/store/useSettingsStore";
import { useUploadStore } from "@/store/useUploadStore";
import { useResultStore } from "@/store/useResultStore";
import clsx from "clsx";

const NAV_ITEMS = [
  { label: "Dashboard", icon: LayoutDashboard, active: true },
  { label: "Frame Analysis", icon: Frame, active: false },
  { label: "Mission Control", icon: Radio, active: false },
];

export function Sidebar() {
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen);
  const clearAll = useUploadStore((s) => s.clearAll);
  const clearResult = useResultStore((s) => s.clearResult);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["system-info"],
    queryFn: fetchSystemInfo,
    staleTime: 30_000,
    retry: 1,
  });

  const status = isError ? "OFFLINE" : (data?.status ?? "BUSY");

  return (
    <motion.aside
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={clsx(
        "flex h-full shrink-0 flex-col gap-6 border-4 border-paper bg-ink p-5",
        sidebarOpen ? "block w-full lg:w-[280px]" : "hidden lg:block lg:w-[92px]",
      )}
    >
      <div className="border-b-4 border-paper pb-4">
        <div className="flex items-center gap-2">
          <Satellite className="text-cyan" size={sidebarOpen ? 26 : 24} />
          {sidebarOpen && (
            <h1 className="font-display text-2xl font-black uppercase leading-none tracking-tighter">
              FrameSat AI
            </h1>
          )}
        </div>
        <div className="mt-3">
          <StatusChip
            status={status}
            label={
              isLoading
                ? "CONNECTING"
                : isError
                  ? "BACKEND OFFLINE"
                  : "AI-CORE ACTIVE"
            }
          />
        </div>
      </div>

      {sidebarOpen && (
        <nav>
          <ul className="flex flex-col gap-2">
            {NAV_ITEMS.map(({ label, icon: Icon, active }) => (
              <li key={label}>
                <button
                  className={clsx(
                    "flex w-full items-center gap-3 border-2 p-3 font-display text-sm font-bold uppercase transition-colors",
                    active
                      ? "border-paper bg-paper text-ink shadow-brutalist"
                      : "border-transparent text-paper/70 hover:border-paper hover:text-paper",
                  )}
                  disabled={!active}
                >
                  <Icon size={18} />
                  {label}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      )}

      {sidebarOpen && (
        <div className="flex flex-col gap-3">
          <InfoCard label="Model" value={data?.model ?? "—"} loading={isLoading} />
          <InfoCard label="Backend" value={data?.backend ?? "—"} loading={isLoading} />
          <InfoCard label="Device" value={data?.device ?? "—"} loading={isLoading} />
          <InfoCard label="Version" value={data?.version ?? "—"} loading={isLoading} />
          <InfoCard
            label="Status"
            value={isError ? "OFFLINE" : (data?.status ?? "—")}
            loading={isLoading}
          />
        </div>
      )}

      {sidebarOpen && (
        <button
          onClick={() => {
            clearAll();
            clearResult();
          }}
          className="mt-auto flex items-center justify-center gap-2 border-4 border-alert bg-ink py-3 font-display text-sm font-black uppercase tracking-tight text-alert hover:bg-alert hover:text-ink"
        >
          <RotateCcw size={16} />
          Reset Session
        </button>
      )}
    </motion.aside>
  );
}
