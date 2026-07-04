"use client";

import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { Footer } from "@/components/layout/Footer";
import { UploadWorkspace } from "@/components/upload/UploadWorkspace";
import { SevirWorkspace } from "@/components/sevir/SevirWorkspace";
import { DataSourceTabs } from "@/components/dashboard/DataSourceTabs";
import { RuntimeCards } from "@/components/dashboard/RuntimeCards";
import { QualityMetricsCards } from "@/components/dashboard/QualityMetricsCards";
import { ToastStack } from "@/components/common/ToastStack";
import { useDataSourceStore } from "@/store/useDataSourceStore";
import dynamic from "next/dynamic";

// Result viewer contains large images — lazy load it.
const ResultViewer = dynamic(
  () => import("@/components/viewer/ResultViewer").then((m) => m.ResultViewer),
  {
    loading: () => (
      <div className="h-64 animate-pulse border-4 border-paper bg-ink" />
    ),
    ssr: false,
  },
);

export function Dashboard() {
  const mode = useDataSourceStore((s) => s.mode);

  return (
    <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col gap-6 bg-panel p-5 lg:flex-row">
      <Sidebar />

      <main className="flex-1">
        <Header />

        <div className="flex flex-col gap-6">
          <DataSourceTabs />
          {mode === "upload" ? <UploadWorkspace /> : <SevirWorkspace />}
          <ResultViewer />
          <QualityMetricsCards />
          <RuntimeCards />
        </div>

        <Footer />
      </main>

      <ToastStack />
    </div>
  );
}
