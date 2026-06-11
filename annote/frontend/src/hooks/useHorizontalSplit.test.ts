import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRef } from "react";

import { useHorizontalSplit } from "./useHorizontalSplit";

describe("useHorizontalSplit", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe = vi.fn();
        disconnect = vi.fn();
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });
  it("initializes trailing width when enabled", () => {
    const containerRef = createRef<HTMLDivElement>();
    const el = document.createElement("div");
    Object.defineProperty(el, "clientWidth", { value: 1000, configurable: true });
    containerRef.current = el;

    const { result } = renderHook(() =>
      useHorizontalSplit(containerRef, { enabled: true, defaultTrailingRatio: 0.4 }),
    );

    expect(result.current.trailingWidth).toBe(400);
  });

  it("resets trailing width on double-click handler", () => {
    const containerRef = createRef<HTMLDivElement>();
    const el = document.createElement("div");
    Object.defineProperty(el, "clientWidth", { value: 1000, configurable: true });
    containerRef.current = el;

    const { result } = renderHook(() =>
      useHorizontalSplit(containerRef, { enabled: true, defaultTrailingRatio: 0.5 }),
    );

    act(() => {
      result.current.dividerProps.onDoubleClick();
    });

    expect(result.current.trailingWidth).toBe(500);
  });
});
