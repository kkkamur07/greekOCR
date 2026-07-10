import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  API_BASE_URL: "https://api.nomicous.com",
  API_ORIGIN: "https://api.nomicous.com",
  fetchBinaryApi: vi.fn(),
}));

vi.mock("../auth/storage", () => ({
  getAccessToken: () => "memory-only-token",
}));

import { resolveProtectedMediaUrl } from "./AuthenticatedImage";

describe("AuthenticatedImage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("rejects a protected-media URL outside the configured API origin", () => {
    expect(
      resolveProtectedMediaUrl("https://attacker.example/image.png"),
    ).toBeNull();
    expect(
      resolveProtectedMediaUrl("http://api.nomicous.com/image.png"),
    ).toBeNull();
  });

  it("uses only document-part media paths", () => {
    expect(resolveProtectedMediaUrl("/media/parts/part-1?w=200")).toBe(
      "https://api.nomicous.com/media/parts/part-1?w=200",
    );
    expect(resolveProtectedMediaUrl("/documents/part-1")).toBeNull();
  });
});
