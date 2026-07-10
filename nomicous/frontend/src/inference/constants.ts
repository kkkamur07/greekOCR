export const HELPER_BASE_URL =
  process.env.NEXT_PUBLIC_INFERENCE_HELPER_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8001";

export const HELPER_PROBE_TIMEOUT_MS = 2_000;

export const INFERENCE_HELPER_RELEASE_TAG = "inference-helper-v0.1.0";
export const INFERENCE_HELPER_RELEASES_URL =
  "https://github.com/kkkamur07/greekOCR/releases/tag/inference-helper-v0.1.0";

const INFERENCE_HELPER_DOWNLOAD_BASE =
  "https://github.com/kkkamur07/greekOCR/releases/download/inference-helper-v0.1.0";

export const INFERENCE_HELPER_MACOS_DMG_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-macos.dmg`;
export const INFERENCE_HELPER_WINDOWS_ZIP_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-windows.zip`;
export const INFERENCE_HELPER_LINUX_TARBALL_URL = `${INFERENCE_HELPER_DOWNLOAD_BASE}/nomicous-inference-helper-linux.tar.gz`;

export const DEFAULT_SEGMENT_REGISTRY_MODEL_ID = "greek-kraken-segment-v1";
