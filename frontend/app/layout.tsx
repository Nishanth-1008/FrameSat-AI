import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "FrameSat AI — Temporal Interpolation Engine",
  description:
    "AI-powered satellite frame interpolation dashboard built on Practical-RIFE.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-panel text-paper antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
