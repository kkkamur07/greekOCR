import { afterEach, describe, expect, it, vi } from "vitest";

import { HELPER_BASE_URL, HELPER_INFO_PATH } from "./constants";
import {
  fetchHelperInfo,
  isModelLocalEligible,
  isModelRemoteOnly,
  modelCacheState,
  parseHelperInfo,
  sameHelperModels,
  shouldRunOnLocalHelper,
  type HelperModelInfo,
} from "./helperInfo";

const HELPER_BODY = {
  service: "nomicous-inference-helper",
  version: "1.4.2",
  models: [
    {
      registry_model_id: "blla-segment",
      task: "segment",
      host_eligibility: "local",
      tags: ["stable"],
      cached: true,
    },
  ],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const models: HelperModelInfo[] = [
  {
    registry_model_id: "greek-calamari-v1",
    task: "transcribe",
    host_eligibility: "local",
    tags: ["stable"],
    cached: true,
  },
  {
    registry_model_id: "blla-segment",
    task: "segment",
    host_eligibility: "local",
    tags: ["stable"],
    cached: false,
  },
  {
    registry_model_id: "flexible-model",
    task: "transcribe",
    host_eligibility: "any",
    tags: ["stable"],
    cached: true,
  },
  {
    registry_model_id: "future-cloud-model",
    task: "transcribe",
    host_eligibility: "remote",
    tags: ["stable"],
    cached: false,
  },
];

describe("fetchHelperInfo", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("accepts a response that identifies itself as the helper", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(HELPER_BODY));
    vi.stubGlobal("fetch", fetchMock);

    const info = await fetchHelperInfo();

    expect(info?.version).toBe("1.4.2");
    expect(info?.models).toHaveLength(1);
    expect(info?.models[0].cached).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      `${HELPER_BASE_URL}${HELPER_INFO_PATH}`,
      expect.objectContaining({
        method: "GET",
        targetAddressSpace: "loopback",
      }),
    );
  });

  it("treats an HTTP 200 without a service field as helper absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ version: "1.0.0", models: [] })),
    );

    await expect(fetchHelperInfo()).resolves.toBeNull();
  });

  it("treats a foreign process answering 200 as helper absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ service: "some-dev-server" })),
    );

    await expect(fetchHelperInfo()).resolves.toBeNull();
  });

  it("treats a non-200 response as helper absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(HELPER_BODY, 404)),
    );

    await expect(fetchHelperInfo()).resolves.toBeNull();
  });

  it("treats a non-JSON body as helper absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response("<html>not me</html>", { status: 200 }),
        ),
    );

    await expect(fetchHelperInfo()).resolves.toBeNull();
  });

  it("treats an unreachable port as helper absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    await expect(fetchHelperInfo()).resolves.toBeNull();
  });
});

describe("parseHelperInfo", () => {
  it("drops malformed model entries but keeps the document", () => {
    const info = parseHelperInfo({
      service: "nomicous-inference-helper",
      version: "2.0.0",
      models: [null, { task: "segment" }, HELPER_BODY.models[0]],
    });

    expect(info?.models.map((model) => model.registry_model_id)).toEqual([
      "blla-segment",
    ]);
  });

  it("rejects anything that is not an object", () => {
    expect(parseHelperInfo(null)).toBeNull();
    expect(parseHelperInfo("nomicous-inference-helper")).toBeNull();
    expect(parseHelperInfo(42)).toBeNull();
  });
});

describe("helper model eligibility", () => {
  it("treats local and any models as local-eligible", () => {
    expect(isModelLocalEligible(models, "greek-calamari-v1")).toBe(true);
    expect(isModelLocalEligible(models, "flexible-model")).toBe(true);
    expect(isModelLocalEligible(models, "missing-model")).toBe(false);
  });

  it("flags remote-only models", () => {
    expect(isModelRemoteOnly(models, "future-cloud-model")).toBe(true);
    expect(isModelRemoteOnly(models, "greek-calamari-v1")).toBe(false);
  });

  it("reports cache state, and null when the model is unknown", () => {
    expect(modelCacheState(models, "greek-calamari-v1")).toBe(true);
    expect(modelCacheState(models, "blla-segment")).toBe(false);
    expect(modelCacheState(models, "missing-model")).toBeNull();
  });
});

describe("shouldRunOnLocalHelper", () => {
  it("uses the helper under automatic and local-only routing", () => {
    for (const routing of ["auto", "local-only"] as const) {
      expect(
        shouldRunOnLocalHelper(models, "blla-segment", {
          helperAvailable: true,
          routing,
        }),
      ).toBe(true);
    }
  });

  it("never contacts the helper under cloud-only routing", () => {
    expect(
      shouldRunOnLocalHelper(models, "blla-segment", {
        helperAvailable: true,
        routing: "cloud-only",
      }),
    ).toBe(false);
  });

  it("never uses the helper when it is down or the model is remote-only", () => {
    expect(
      shouldRunOnLocalHelper(models, "blla-segment", {
        helperAvailable: false,
        routing: "auto",
      }),
    ).toBe(false);
    expect(
      shouldRunOnLocalHelper(models, "future-cloud-model", {
        helperAvailable: true,
        routing: "auto",
      }),
    ).toBe(false);
    expect(
      shouldRunOnLocalHelper(models, "model-the-helper-never-listed", {
        helperAvailable: true,
        routing: "local-only",
      }),
    ).toBe(false);
  });
});

describe("sameHelperModels", () => {
  it("reports equal content across separately parsed documents", () => {
    const left = parseHelperInfo(HELPER_BODY)!.models;
    const right = parseHelperInfo(HELPER_BODY)!.models;
    expect(left).not.toBe(right);
    expect(sameHelperModels(left, right)).toBe(true);
  });

  it("notices a changed cache flag", () => {
    const left = parseHelperInfo(HELPER_BODY)!.models;
    const right = parseHelperInfo({
      ...HELPER_BODY,
      models: [{ ...HELPER_BODY.models[0], cached: false }],
    })!.models;
    expect(sameHelperModels(left, right)).toBe(false);
  });
});
