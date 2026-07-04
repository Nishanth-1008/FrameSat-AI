"use client";

import { motion } from "framer-motion";
import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "md" | "lg";
}

/**
 * Brutalist CTA: thick border, hard offset shadow that collapses on press,
 * uppercase Space Grotesk label.
 */
export function Button({
  variant = "primary",
  size = "md",
  className,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <motion.button
      whileTap={disabled ? undefined : { x: 4, y: 4 }}
      disabled={disabled}
      className={clsx(
        "border-4 font-display font-black uppercase tracking-tight transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40",
        size === "lg" ? "px-8 py-5 text-xl" : "px-5 py-2.5 text-sm",
        variant === "primary" &&
          "border-ink bg-green text-ink shadow-brutalist-ink hover:bg-ink hover:text-green active:shadow-none",
        variant === "secondary" &&
          "border-paper bg-ink text-paper shadow-brutalist hover:bg-paper hover:text-ink active:shadow-none",
        variant === "ghost" &&
          "border-transparent text-muted hover:text-paper hover:border-paper",
        className,
      )}
      {...props}
    >
      {children}
    </motion.button>
  );
}
