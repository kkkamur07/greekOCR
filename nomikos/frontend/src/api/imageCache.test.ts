import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./client", () => ({
  API_BASE_URL: "https://api.nomikos.app",
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

  /**
   * A document page list mounts every thumbnail at once. When there are more
   * pages than the cache bound, the sweep used to run while the callers that
   * had just requested those images were still one microtask away from taking
   * their reference, so it revoked object URLs that were about to be shown.
   * The symptom was a broken-image glyph on an arbitrary subset of the grid,
   * only after a reload, and only on documents longer than the bound.
   */
  it("keeps every concurrently requested image alive past the cache bound", async () => {
    let issued = 0;
    createObjectURL.mockImplementation(() => `blob:image-${(issued += 1)}`);
    const total = MAX_CACHED_PART_IMAGES + 6;

    const images = await Promise.all(
      Array.from({ length: total }, (_unused, index) =>
        acquirePartImage(`/media/parts/part-${index}`),
      ),
    );

    expect(images).toHaveLength(total);
    const shown = new Set(images.map((image) => image.objectUrl));
    // Every image got its own URL, so a revoke can be attributed to one image.
    expect(shown.size).toBe(total);
    for (const call of revokeObjectURL.mock.calls) {
      expect(shown.has(call[0] as string)).toBe(false);
    }

    // Releasing them all lets the cache fall back inside its bound, which is
    // what stops the fix from turning into an unbounded cache.
    for (const image of images) image.release();
    expect(revokeObjectURL).toHaveBeenCalled();
  });

  /**
   * The same invariant with the timing pinned down rather than left to the
   * scheduler. Every request is resolved before any caller gets to run its
   * continuation, so the first entry inserted is both the oldest in LRU order
   * and still unreferenced when the last insertion triggers the sweep. That is
   * exactly the state the claim exists to cover, and without it this evicts
   * the very image the first caller is about to display.
   */
  it("evicts nothing while its callers are still waiting to take a reference", async () => {
    let issued = 0;
    createObjectURL.mockImplementation(() => `blob:image-${(issued += 1)}`);
    const total = MAX_CACHED_PART_IMAGES + 4;

    const resolvers: ((blob: Blob) => void)[] = [];
    vi.mocked(fetchBinaryApi).mockImplementation(
      () =>
        new Promise<Blob>((resolve) => {
          resolvers.push(resolve);
        }),
    );

    const acquisitions = Array.from({ length: total }, (_unused, index) =>
      acquirePartImage(`/media/parts/part-${index}`),
    );
    // Let the dynamic client import settle so every request is actually in
    // flight and has registered its resolver.
    await vi.waitFor(() => expect(resolvers).toHaveLength(total));

    for (const resolve of resolvers) resolve(new Blob(["image"]));
    const images = await Promise.all(acquisitions);

    const shown = new Set(images.map((image) => image.objectUrl));
    expect(shown.size).toBe(total);
    for (const call of revokeObjectURL.mock.calls) {
      expect(shown.has(call[0] as string)).toBe(false);
    }
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
