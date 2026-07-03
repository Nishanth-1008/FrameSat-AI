"use client";

import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { Footer } from "@/components/layout/Footer";
import { UploadWorkspace } from "@/components/upload/UploadWorkspace";
import { RuntimeCards } from "@/components/dashboard/RuntimeCards";
import { ToastStack } from "@/components/common/ToastStack";
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
  return (
    <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col gap-6 bg-panel p-5 lg:flex-row">
      <Sidebar />

      <main className="flex-1">
        <Header />

        <div className="flex flex-col gap-6">
          <UploadWorkspace />
          <ResultViewer />
          <RuntimeCards />
        </div>

        <Footer />
      </main>

      <ToastStack />
    </div>
  );
}
