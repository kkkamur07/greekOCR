import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ZOOM_SETTLE_MS, useZoomLayerHint } from "./zoomLayerHint";

describe("useZoomLayerHint", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("is off until the view moves", () => {
    const { result } = renderHook(() => useZoomLayerHint());
    expect(result.current.transforming).toBe(false);
  });

  it("turns on for a transform and off once the view settles", () => {
    const { result } = renderHook(() => useZoomLayerHint());

    act(() => result.current.markTransforming());
    expect(result.current.transforming).toBe(true);

    act(() => {
      vi.advanceTimersByTime(ZOOM_SETTLE_MS - 1);
    });
    expect(result.current.transforming).toBe(true);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(result.current.transforming).toBe(false);
  });

  it("stays on across the frames of a continuous gesture", () => {
    const { result } = renderHook(() => useZoomLayerHint());

    for (let frame = 0; frame < 20; frame += 1) {
      act(() => result.current.markTransforming());
      act(() => {
        vi.advanceTimersByTime(16);
      });
      expect(result.current.transforming).toBe(true);
    }

    act(() => {
      vi.advanceTimersByTime(ZOOM_SETTLE_MS);
    });
    expect(result.current.transforming).toBe(false);
  });

  it("does not leave a timer running after unmount", () => {
    const { result, unmount } = renderHook(() => useZoomLayerHint());
    act(() => result.current.markTransforming());
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
