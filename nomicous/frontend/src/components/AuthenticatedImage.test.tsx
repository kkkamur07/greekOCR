import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  API_BASE_URL: "https://api.nomicous.com",
  fetchBinaryApi: vi.fn(),
}));

// The `../auth/storage` mock that used to sit here was dead: this module reads no
// token. It was also load-bearing by accident, stubbing out `setAccessToken`/
// `clearAccessToken` for anything that imported storage transitively.

import { resolveProtectedMediaUrl } from "./AuthenticatedImage";

describe("resolveProtectedMediaUrl", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("rejects a foreign origin", () => {
    expect(
      resolveProtectedMediaUrl("https://attacker.example/image.png"),
    ).toBeNull();
  });

  it("uses only document-part media paths", () => {
    expect(resolveProtectedMediaUrl("/media/parts/part-1?w=200")).toBe(
      "https://api.nomicous.com/media/parts/part-1?w=200",
    );
    expect(resolveProtectedMediaUrl("/documents/part-1")).toBeNull();
  });

  it("rejects a width that is not a plain positive integer", () => {
    expect(resolveProtectedMediaUrl("/media/parts/part-1?w=0")).toBeNull();
    expect(resolveProtectedMediaUrl("/media/parts/part-1?w=-5")).toBeNull();
    expect(resolveProtectedMediaUrl("/media/parts/part-1?w=200&x=1")).toBeNull();
    expect(resolveProtectedMediaUrl("/media/parts/part-1?other=1")).toBeNull();
  });
});

// The plaintext-http guard lives in `src/api/imageCache.ts`, which computes its own
// origin from `process.env.NEXT_PUBLIC_API_BASE_URL` at module load and never reads
// `API_BASE_URL` from `../api/client`. Mocking that export therefore proved nothing
// about the guard: `http://api.nomicous.com/...` was rejected merely for being a
// different origin than the default `http://localhost:8000`, and the https rule could
// be deleted outright with this file still green.
//
// Driving the real environment variable is the only way to put the configured API on a
// non-localhost host and see which protocols survive.
describe("protected media over plaintext http", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  async function normalizeWithApiBase(apiBase: string, src: string) {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", apiBase);
    const { normalizePartImagePath } = await import("../api/imageCache");
    return normalizePartImagePath(src);
  }

  it("refuses a page image when the API itself is plain http on a real host", async () => {
    expect(
      await normalizeWithApiBase(
        "http://api.nomicous.com",
        "http://api.nomicous.com/media/parts/part-1",
      ),
    ).toBeNull();
  });

  it("allows the same path when the API is https", async () => {
    expect(
      await normalizeWithApiBase(
        "https://api.nomicous.com",
        "https://api.nomicous.com/media/parts/part-1",
      ),
    ).toBe("/media/parts/part-1");
  });

  it("still allows plain http against localhost, so development works", async () => {
    expect(
      await normalizeWithApiBase(
        "http://localhost:8000",
        "http://localhost:8000/media/parts/part-1",
      ),
    ).toBe("/media/parts/part-1");
  });
});
