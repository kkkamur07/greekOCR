import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  isModelLocalEligible,
  isModelRemoteOnly,
  shouldRunOnLocalHelper,
} from "./catalog";
import type { HelperCatalogModel } from "./catalog";
import {
  loadInferencePreference,
  preferCloudInference,
  saveInferencePreference,
} from "./preference";
import { probeHelperHealth } from "./probe";

const catalog: HelperCatalogModel[] = [
  {
    registry_model_id: "greek-calamari-v1",
    task: "transcribe",
    architecture: "calamari",
    device: "cpu",
    host_eligibility: "local",
    tags: ["stable"],
  },
  {
    registry_model_id: "blla-segment",
    task: "segment",
    architecture: "blla",
    device: "cpu",
    host_eligibility: "local",
    tags: ["stable"],
  },
  {
    registry_model_id: "flexible-model",
    task: "transcribe",
    architecture: "calamari",
    device: "cpu",
    host_eligibility: "any",
    tags: ["stable"],
  },
  {
    registry_model_id: "future-cloud-model",
    task: "transcribe",
    architecture: "calamari",
    device: "cuda",
    host_eligibility: "remote",
    tags: ["stable"],
  },
];

describe("inference preference", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("defaults to local preference", () => {
    expect(loadInferencePreference()).toBe("local");
    expect(preferCloudInference()).toBe(false);
  });

  it("persists cloud preference across reloads", () => {
    saveInferencePreference("cloud");
    expect(loadInferencePreference()).toBe("cloud");
    expect(preferCloudInference()).toBe(true);
  });
});

describe("helper catalog eligibility", () => {
  it("treats local and any models as local-eligible", () => {
    expect(isModelLocalEligible(catalog, "greek-calamari-v1")).toBe(true);
    expect(isModelLocalEligible(catalog, "missing-model")).toBe(false);
  });

  it("flags remote-only models", () => {
    expect(isModelRemoteOnly(catalog, "future-cloud-model")).toBe(true);
    expect(isModelRemoteOnly(catalog, "greek-calamari-v1")).toBe(false);
  });
});

describe("shouldRunOnLocalHelper", () => {
  it("uses the helper for local-only models even when cloud is preferred", () => {
    expect(
      shouldRunOnLocalHelper(catalog, "blla-segment", {
        helperAvailable: true,
        preferCloud: true,
      }),
    ).toBe(true);
  });

  it("honors cloud preference for any-host models", () => {
    expect(
      shouldRunOnLocalHelper(catalog, "flexible-model", {
        helperAvailable: true,
        preferCloud: true,
      }),
    ).toBe(false);
    expect(
      shouldRunOnLocalHelper(catalog, "flexible-model", {
        helperAvailable: true,
        preferCloud: false,
      }),
    ).toBe(true);
  });

  it("never uses the helper when it is down or the model is remote-only", () => {
    expect(
      shouldRunOnLocalHelper(catalog, "blla-segment", {
        helperAvailable: false,
        preferCloud: false,
      }),
    ).toBe(false);
    expect(
      shouldRunOnLocalHelper(catalog, "future-cloud-model", {
        helperAvailable: true,
        preferCloud: false,
      }),
    ).toBe(false);
  });
});

describe("probeHelperHealth", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns false when helper is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );
    await expect(probeHelperHealth()).resolves.toBe(false);
  });

  it("returns true when health endpoint responds ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok" }),
      }),
    );
    await expect(probeHelperHealth()).resolves.toBe(true);
  });
});
