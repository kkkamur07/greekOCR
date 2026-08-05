import { describe, expect, it } from "vitest";

import {
  HELPER_BASE_URL,
  HELPER_INFO_PATH,
  HELPER_PROBE_TIMEOUT_MS,
  HELPER_SERVICE_NAME,
  INFERENCE_HELPER_MACOS_INTEL_DMG_URL,
  INFERENCE_HELPER_LINUX_TARBALL_URL,
  INFERENCE_HELPER_MACOS_DMG_URL,
  INFERENCE_HELPER_RELEASES_URL,
  INFERENCE_HELPER_WINDOWS_ZIP_URL,
} from "./constants";

describe("inference helper download constants", () => {
  it("points releases and assets at GitHub releases/latest", () => {
    expect(INFERENCE_HELPER_RELEASES_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest",
    );
    expect(INFERENCE_HELPER_MACOS_INTEL_DMG_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-macos-intel.dmg",
    );
    expect(INFERENCE_HELPER_MACOS_DMG_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-macos.dmg",
    );
    expect(INFERENCE_HELPER_WINDOWS_ZIP_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-windows.zip",
    );
    expect(INFERENCE_HELPER_LINUX_TARBALL_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-linux.tar.gz",
    );
    expect(HELPER_PROBE_TIMEOUT_MS).toBeGreaterThan(0);
  });
});

describe("inference helper discovery constants", () => {
  it("uses one IPv4 loopback origin, not a list of candidates", () => {
    expect(HELPER_BASE_URL).toBe("http://127.0.0.1:8001");
    expect(HELPER_INFO_PATH).toBe("/inference/v1/info");
  });

  it("pins the identity a responder must claim to be trusted", () => {
    expect(HELPER_SERVICE_NAME).toBe("nomicous-inference-helper");
  });
});
