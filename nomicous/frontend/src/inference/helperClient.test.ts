import { afterEach, describe, expect, it, vi } from "vitest";

import { HELPER_BASE_URLS } from "./constants";
import { fetchHelper } from "./helperClient";

describe("fetchHelper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("tries localhost after connection failures on earlier loopback URLs", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("connection refused"))
      .mockResolvedValueOnce(new Response("ok", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await fetchHelper("/health");

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      `${HELPER_BASE_URLS[0]}/health`,
      { targetAddressSpace: "loopback" },
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      `${HELPER_BASE_URLS[1]}/health`,
      { targetAddressSpace: "loopback" },
    );
  });

  it("does not retry an aborted request", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new DOMException("aborted", "AbortError"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      fetchHelper("/health", { signal: controller.signal }),
    ).rejects.toThrow("aborted");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
