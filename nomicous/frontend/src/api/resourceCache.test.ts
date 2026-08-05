import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearResourceCache,
  invalidateTags,
  patchResource,
  peekResource,
  readResource,
  subscribeToResource,
} from "./resourceCache";

afterEach(() => {
  clearResourceCache();
});

describe("readResource", () => {
  it("shares one in-flight request between concurrent readers", async () => {
    const read = vi.fn(async () => "value");

    const [first, second] = await Promise.all([
      readResource(["thing", 1], ["thing"], read),
      readResource(["thing", 1], ["thing"], read),
    ]);

    expect(first).toBe("value");
    expect(second).toBe("value");
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("serves a settled read again without going back to the server", async () => {
    const read = vi.fn(async () => "value");

    await readResource(["thing"], ["thing"], read);
    expect(await readResource(["thing"], ["thing"], read)).toBe("value");

    expect(read).toHaveBeenCalledTimes(1);
  });

  it("keys reads apart so one resource cannot answer for another", async () => {
    const read = vi.fn(async () => "value");

    await readResource(["thing", "a"], ["thing"], read);
    await readResource(["thing", "b"], ["thing"], read);

    expect(read).toHaveBeenCalledTimes(2);
  });

  it("does not retain a failure, so the next reader may try again", async () => {
    const read = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce("value");

    await expect(readResource(["thing"], ["thing"], read)).rejects.toThrow(
      "boom",
    );
    expect(await readResource(["thing"], ["thing"], read)).toBe("value");
  });

  it("goes back to the server when forced past the freshness window", async () => {
    const read = vi
      .fn()
      .mockResolvedValueOnce("first")
      .mockResolvedValueOnce("second");

    await readResource(["thing"], ["thing"], read);
    const forced = await readResource(["thing"], ["thing"], read, {
      force: true,
    });

    expect(forced).toBe("second");
  });

  it("shares an in-flight request even when forced, rather than duplicating it", async () => {
    const read = vi.fn(
      () => new Promise<string>((resolve) => setTimeout(() => resolve("v"), 5)),
    );

    const [a, b] = await Promise.all([
      readResource(["thing"], ["thing"], read),
      readResource(["thing"], ["thing"], read, { force: true }),
    ]);

    expect([a, b]).toEqual(["v", "v"]);
    expect(read).toHaveBeenCalledTimes(1);
  });
});

describe("invalidateTags", () => {
  it("drops every read carrying the tag, whatever its key", async () => {
    const read = vi.fn(async () => "value");
    await readResource(["dashboard", "p1"], ["project:p1"], read);
    await readResource(["editor", "p1"], ["project:p1"], read);
    expect(read).toHaveBeenCalledTimes(2);

    invalidateTags(["project:p1"]);

    await readResource(["dashboard", "p1"], ["project:p1"], read);
    await readResource(["editor", "p1"], ["project:p1"], read);
    expect(read).toHaveBeenCalledTimes(4);
  });

  it("leaves reads that do not carry the tag alone", async () => {
    const read = vi.fn(async () => "value");
    await readResource(["other"], ["projects"], read);

    invalidateTags(["project:p1"]);

    await readResource(["other"], ["projects"], read);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("tells a subscriber that the read it is showing went stale", async () => {
    const onInvalidated = vi.fn();
    await readResource(["dashboard"], ["projects"], async () => "value");
    const unsubscribe = subscribeToResource(["dashboard"], onInvalidated);

    invalidateTags(["projects"]);
    expect(onInvalidated).toHaveBeenCalledTimes(1);

    unsubscribe();
    await readResource(["dashboard"], ["projects"], async () => "value");
    invalidateTags(["projects"]);
    expect(onInvalidated).toHaveBeenCalledTimes(1);
  });

  it("does not let a request that was in flight when it was invalidated repopulate the cache", async () => {
    let release!: (value: string) => void;
    const pending = readResource(
      ["thing"],
      ["thing"],
      () =>
        new Promise<string>((resolve) => {
          release = resolve;
        }),
    );

    invalidateTags(["thing"]);
    release("stale");
    await pending;

    expect(peekResource(["thing"])).toBeNull();
  });
});

describe("patchResource", () => {
  it("folds a mutation's own response into a cached read", async () => {
    await readResource(["doc"], ["doc"], async () => ({ name: "before" }));

    patchResource(["doc"], { name: "after" });

    expect(peekResource<{ name: string }>(["doc"])?.data).toEqual({
      name: "after",
    });
  });

  it("does not resurrect a read that was invalidated", async () => {
    await readResource(["doc"], ["doc"], async () => ({ name: "before" }));
    invalidateTags(["doc"]);

    patchResource(["doc"], { name: "after" });

    expect(peekResource(["doc"])).toBeNull();
  });
});
