import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  cloudInferenceEnabled,
  loadInferenceRouting,
  localInferenceEnabled,
  normalizeInferenceRouting,
  saveInferenceRouting,
} from "./preference";

describe("inference routing preference", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("defaults to automatic routing", () => {
    expect(loadInferenceRouting()).toBe("auto");
  });

  it("persists each of the three states across reloads", () => {
    for (const routing of ["auto", "local-only", "cloud-only"] as const) {
      saveInferenceRouting(routing);
      expect(loadInferenceRouting()).toBe(routing);
    }
  });

  it("migrates the legacy binary preference", () => {
    localStorage.setItem("nomicous_inference_preference", "cloud");
    expect(loadInferenceRouting()).toBe("cloud-only");

    localStorage.setItem("nomicous_inference_preference", "local");
    expect(loadInferenceRouting()).toBe("auto");
  });

  it("falls back to automatic for unknown stored values", () => {
    expect(normalizeInferenceRouting(null)).toBe("auto");
    expect(normalizeInferenceRouting("")).toBe("auto");
    expect(normalizeInferenceRouting("gpu-farm")).toBe("auto");
  });

  it("disables cloud only under local-only, and local only under cloud-only", () => {
    expect(cloudInferenceEnabled("auto")).toBe(true);
    expect(cloudInferenceEnabled("cloud-only")).toBe(true);
    expect(cloudInferenceEnabled("local-only")).toBe(false);

    expect(localInferenceEnabled("auto")).toBe(true);
    expect(localInferenceEnabled("local-only")).toBe(true);
    expect(localInferenceEnabled("cloud-only")).toBe(false);
  });
});
