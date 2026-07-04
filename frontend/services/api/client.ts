import type { ApiErrorShape } from "@/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error implements ApiErrorShape {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Thin wrapper around fetch. All backend calls must go through here
 * (or a service built on top of it) — never call fetch directly from
 * components.
 */
export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "Unable to reach the FrameSat backend. Check your connection or try again shortly.",
    );
  }

  if (!response.ok) {
    let message = `Backend responded with status ${response.status}.`;

    try {
      const body = await response.json();
      // Never surface raw tracebacks — only a short backend-provided message.
      if (typeof body?.detail === "string") {
        message = body.detail;
      } else if (typeof body?.message === "string") {
        message = body.message;
      }
    } catch {
      // ignore JSON parse failures, keep default message
    }

    throw new ApiError(message, response.status);
  }

  return response.json() as Promise<T>;
}
