import { afterEach, describe, expect, it, vi } from "vitest";

import { HELPER_BASE_URL, HELPER_INFO_PATH } from "./constants";
import { fetchHelper } from "./helperClient";

describe("fetchHelper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("calls the single loopback origin and marks the request as loopback", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response("ok", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await fetchHelper(HELPER_INFO_PATH);

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      `${HELPER_BASE_URL}${HELPER_INFO_PATH}`,
      { targetAddressSpace: "loopback" },
    );
  });

  it("does not walk alternative URLs when the helper is unreachable", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new TypeError("connection refused"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchHelper(HELPER_INFO_PATH)).rejects.toThrow(
      "connection refused",
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not retry an aborted request", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new DOMException("aborted", "AbortError"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      fetchHelper(HELPER_INFO_PATH, { signal: controller.signal }),
    ).rejects.toThrow("aborted");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
