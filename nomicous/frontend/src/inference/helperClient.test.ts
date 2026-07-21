import { afterEach, describe, expect, it, vi } from "vitest";

import { HELPER_BASE_URLS } from "./constants";
import { fetchHelper } from "./helperClient";

describe("fetchHelper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("tries the IPv6 and localhost fallbacks after connection failures", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("connection refused"))
      .mockRejectedValueOnce(new TypeError("connection refused"))
      .mockResolvedValueOnce(new Response("ok", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await fetchHelper("/health");

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      `${HELPER_BASE_URLS[0]}/health`,
      undefined,
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      `${HELPER_BASE_URLS[2]}/health`,
      undefined,
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
