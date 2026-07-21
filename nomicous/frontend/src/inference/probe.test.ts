import { afterEach, describe, expect, it, vi } from "vitest";

import { HELPER_BASE_URLS } from "./constants";
import { probeHelperHealth } from "./probe";

describe("probeHelperHealth", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("finds a healthy helper on a fallback loopback address", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("connection refused"))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ status: "ok" }), { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(probeHelperHealth()).resolves.toBe(true);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      `${HELPER_BASE_URLS[0]}/health`,
      expect.objectContaining({
        method: "GET",
        targetAddressSpace: "loopback",
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      `${HELPER_BASE_URLS[1]}/health`,
      expect.objectContaining({
        method: "GET",
        targetAddressSpace: "loopback",
      }),
    );
  });
});
