const DEFAULT_HELPER_BASE_URL = "http://127.0.0.1:8001";
const configuredHelperBaseUrl = process.env.NEXT_PUBLIC_INFERENCE_HELPER_URL
  ?.trim()
  .replace(/\/+$/, "");

export const HELPER_BASE_URL =
  configuredHelperBaseUrl || DEFAULT_HELPER_BASE_URL;

/** All loopback hosts the browser may use to reach the local helper. */
export const HELPER_LOOPBACK_BASE_URLS = [
  "http://127.0.0.1:8001",
  "http://localhost:8001",
  "http://[::1]:8001",
] as const;

// Try configured URL first, then every IPv4 / IPv6 / localhost loopback form.
export const HELPER_BASE_URLS = [
  HELPER_BASE_URL,
  ...HELPER_LOOPBACK_BASE_URLS,
].filter((url, index, urls) => urls.indexOf(url) === index);

export const HELPER_PROBE_TIMEOUT_MS = 2_000;
/** How often the page editor re-probes for a newly started helper. */
export const HELPER_PROBE_INTERVAL_MS = 5_000;

export const INFERENCE_HELPER_RELEASES_URL =
  "https://github.com/kkkamur07/greekOCR/releases/latest";

const INFERENCE_HELPER_DOWNLOAD_BASE =
  "https://github.com/kkkamur07/greekOCR/releases/latest/download";

export const INFERENCE_HELPER_MACOS_INTEL_DMG_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-macos-intel.dmg`;
/** Apple-silicon download keeps the established release asset name. */
export const INFERENCE_HELPER_MACOS_DMG_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-macos.dmg`;
export const INFERENCE_HELPER_WINDOWS_ZIP_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-windows.zip`;
export const INFERENCE_HELPER_LINUX_TARBALL_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-linux.tar.gz`;

export const DEFAULT_SEGMENT_REGISTRY_MODEL_ID = "blla-segment";
