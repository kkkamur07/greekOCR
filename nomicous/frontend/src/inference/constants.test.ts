import { describe, expect, it } from "vitest";

import {
  HELPER_PROBE_INTERVAL_MS,
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
    expect(INFERENCE_HELPER_MACOS_DMG_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-macos.dmg",
    );
    expect(INFERENCE_HELPER_WINDOWS_ZIP_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-windows.zip",
    );
    expect(INFERENCE_HELPER_LINUX_TARBALL_URL).toBe(
      "https://github.com/kkkamur07/greekOCR/releases/latest/download/nomicous-inference-helper-linux.tar.gz",
    );
    expect(HELPER_PROBE_INTERVAL_MS).toBeGreaterThan(0);
  });
});
