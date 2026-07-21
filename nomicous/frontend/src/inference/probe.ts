import { HELPER_BASE_URLS, HELPER_PROBE_TIMEOUT_MS } from "./constants";

export async function probeHelperHealth(): Promise<boolean> {
  for (const baseUrl of HELPER_BASE_URLS) {
    const controller = new AbortController();
    const timeout = window.setTimeout(
      () => controller.abort(),
      HELPER_PROBE_TIMEOUT_MS,
    );
    try {
      const response = await fetch(`${baseUrl}/health`, {
        method: "GET",
        signal: controller.signal,
      });
      if (!response.ok) continue;
      const body = (await response.json()) as { status?: string };
      if (body.status === "ok") return true;
    } catch {
      // Try the next loopback address.
    } finally {
      window.clearTimeout(timeout);
    }
  }
  return false;
}
