import { afterEach, describe, expect, it, vi } from "vitest";

import { runLocalInference } from "./localClient";

describe("runLocalInference", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("calls the loopback helper without a browser-shipped secret", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        task: "segment",
        output: { blocks: [], lines: [] },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await runLocalInference({
      task: "segment",
      registry_model_id: "greek-kraken-segment-v1",
      image_bytes: "encoded-image",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8001/inference/v1/run",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });
});
