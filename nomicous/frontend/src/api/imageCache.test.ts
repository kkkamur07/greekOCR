import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./client", () => ({
  API_BASE_URL: "https://api.nomicous.com",
  fetchBinaryApi: vi.fn(),
}));

import { fetchBinaryApi } from "./client";
import {
  acquirePartImage,
  clearImageCache,
  fetchPartImage,
  invalidatePartImage,
  MAX_CACHED_PART_IMAGES,
  normalizePartImagePath,
} from "./imageCache";

const NativeURL = URL;

describe("imageCache", () => {
  const createObjectURL = vi.fn(() => "blob:part-image");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    class TestURL extends NativeURL {}
    Object.assign(TestURL, { createObjectURL, revokeObjectURL });
    vi.stubGlobal("URL", TestURL);
    vi.mocked(fetchBinaryApi).mockResolvedValue(new Blob(["image"]));
  });

  afterEach(() => {
    clearImageCache();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("deduplicates concurrent requests for one image representation", async () => {
    await Promise.all([
      fetchPartImage("/media/parts/part-1"),
      fetchPartImage("/media/parts/part-1"),
    ]);

    expect(fetchBinaryApi).toHaveBeenCalledTimes(1);
    expect(fetchBinaryApi).toHaveBeenCalledWith("/media/parts/part-1");
  });

  it("keeps full and thumbnail representations separate", async () => {
    await fetchPartImage("/media/parts/part-1");
    await fetchPartImage("/media/parts/part-1?w=200");

    expect(fetchBinaryApi).toHaveBeenCalledTimes(2);
    expect(fetchBinaryApi).toHaveBeenLastCalledWith(
      "/media/parts/part-1?w=200",
    );
  });

  it("revokes all variants when a part is invalidated", async () => {
    await fetchPartImage("/media/parts/part-1");
    await fetchPartImage("/media/parts/part-1?w=200");

    invalidatePartImage("part-1");

    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
  });

  it("evicts the least recently used image once the bound is exceeded", async () => {
    for (let index = 0; index < MAX_CACHED_PART_IMAGES; index += 1) {
      await fetchPartImage(`/media/parts/part-${index}`);
    }
    // Re-reading part-0 makes part-1 the least recently used one.
    await fetchPartImage("/media/parts/part-0");

    await fetchPartImage("/media/parts/part-overflow");

    expect(revokeObjectURL).toHaveBeenCalledTimes(1);
    vi.mocked(fetchBinaryApi).mockClear();
    await fetchPartImage("/media/parts/part-0");
    expect(fetchBinaryApi).not.toHaveBeenCalled();
    await fetchPartImage("/media/parts/part-1");
    expect(fetchBinaryApi).toHaveBeenCalledWith("/media/parts/part-1");
  });

  it("never evicts an image that is still on screen", async () => {
    const held = await acquirePartImage("/media/parts/part-held");
    for (let index = 0; index < MAX_CACHED_PART_IMAGES * 2; index += 1) {
      await fetchPartImage(`/media/parts/part-${index}`);
    }
    vi.mocked(fetchBinaryApi).mockClear();

    await fetchPartImage("/media/parts/part-held");

    expect(fetchBinaryApi).not.toHaveBeenCalled();

    // Once nothing is showing it any more, it is evictable like anything else.
    held.release();
    for (let index = 0; index < MAX_CACHED_PART_IMAGES; index += 1) {
      await fetchPartImage(`/media/parts/later-${index}`);
    }
    await fetchPartImage("/media/parts/part-held");

    expect(fetchBinaryApi).toHaveBeenCalledWith("/media/parts/part-held");
  });

  it("allows only same-origin part-image URLs", () => {
    expect(normalizePartImagePath("/media/parts/part-1?w=200")).toBe(
      "/media/parts/part-1?w=200",
    );
    expect(
      normalizePartImagePath("https://attacker.example/media/parts/part-1"),
    ).toBeNull();
    expect(normalizePartImagePath("/media/parts/part-1?foo=bar")).toBeNull();
  });
});
