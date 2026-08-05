const DEFAULT_HELPER_BASE_URL = "http://127.0.0.1:8001";
const configuredHelperBaseUrl =
  process.env.NEXT_PUBLIC_INFERENCE_HELPER_URL?.trim().replace(/\/+$/, "");

/**
 * The single loopback origin the browser uses to reach the local helper.
 *
 * Discovery deliberately does not walk a list of candidate URLs. Whatever
 * answers here must still identify itself as the helper (see
 * `fetchHelperInfo`) before we hand it a page image.
 */
export const HELPER_BASE_URL =
  configuredHelperBaseUrl || DEFAULT_HELPER_BASE_URL;

/** Identity the helper must report from `/inference/v1/info`. */
export const HELPER_SERVICE_NAME = "nomicous-inference-helper";

export const HELPER_INFO_PATH = "/inference/v1/info";

export const HELPER_PROBE_TIMEOUT_MS = 2_000;

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
