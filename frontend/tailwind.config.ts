import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brutalist mission-control palette
        ink: "#1a1a1a",       // near-black surface / border
        paper: "#f5f0e8",     // warm off-white (light panels, text-on-dark)
        panel: "#141414",     // page background (darker than ink for depth)
        cyan: "#00ffff",
        green: "#39ff14",
        alert: "#e63b2e",
        muted: "#8f8a80",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        headline: ["'Space Grotesk'", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      borderRadius: {
        DEFAULT: "0px",
        lg: "0px",
        xl: "0px",
        full: "0px",
        pill: "0px",
        card: "0px",
      },
      boxShadow: {
        brutalist: "4px 4px 0px 0px #f5f0e8",
        "brutalist-ink": "4px 4px 0px 0px #1a1a1a",
        "brutalist-lg": "8px 8px 0px 0px #f5f0e8",
        "brutalist-cyan": "4px 4px 0px 0px #00ffff",
        "brutalist-green": "4px 4px 0px 0px #39ff14",
        glowCyan: "0 0 20px rgba(0,255,255,0.35)",
        glowGreen: "0 0 20px rgba(57,255,20,0.35)",
      },
      keyframes: {
        pulseDot: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        scan: {
          "0%": { top: "0%" },
          "100%": { top: "100%" },
        },
      },
      animation: {
        pulseDot: "pulseDot 1.8s ease-in-out infinite",
        scan: "scan 2s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
