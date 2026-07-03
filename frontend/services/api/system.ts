import { apiRequest } from "@/services/api/client";
import type { SystemInfo as _SystemInfo } from "@/types";

export type SystemInfo = _SystemInfo;

/**
 * Calls GET /system to populate the sidebar with real backend
 * metadata (model, device, version, status) instead of hardcoded values.
 */
export async function fetchSystemInfo(): Promise<SystemInfo> {
  return apiRequest<SystemInfo>("/system");
}

export const getSystem = fetchSystemInfo;
