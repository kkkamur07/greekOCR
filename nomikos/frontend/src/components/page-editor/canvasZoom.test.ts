import { describe, expect, it } from "vitest";

import {
  MAX_WHEEL_DELTA,
  PINCH_ZOOM_RATE,
  WHEEL_ZOOM_RATE,
  nextZoomFrameScale,
  wheelDeltaPixels,
  wheelTargetScale,
  zoomAnchoredPosition,
} from "./canvasZoom";

describe("canvasZoom", () => {
  it("zooms by the same ratio per notch at any scale, and clamps", () => {
    const notch = -120;
    const ratioAt = (scale: number) =>
      wheelTargetScale(scale, notch, WHEEL_ZOOM_RATE, 0.15, 8) / scale;
    expect(ratioAt(0.5)).toBeCloseTo(ratioAt(4), 10);
    expect(ratioAt(1)).toBeCloseTo(Math.exp(120 * WHEEL_ZOOM_RATE), 10);
    // Scrolling down zooms out, and both ends stop at the limits.
    expect(wheelTargetScale(1, 120, WHEEL_ZOOM_RATE, 0.15, 8)).toBeLessThan(1);
    expect(wheelTargetScale(7.9, -2000, WHEEL_ZOOM_RATE, 0.15, 8)).toBe(8);
    expect(wheelTargetScale(0.16, 2000, PINCH_ZOOM_RATE, 0.15, 8)).toBe(0.15);
  });

  it("normalises line and page deltas and caps driver flings", () => {
    expect(wheelDeltaPixels(3, 1)).toBe(48);
    expect(wheelDeltaPixels(1, 2)).toBe(MAX_WHEEL_DELTA);
    expect(wheelDeltaPixels(-900, 0)).toBe(-MAX_WHEEL_DELTA);
    expect(wheelDeltaPixels(-7, 0)).toBe(-7);
  });

  it("glides toward the target and arrives exactly", () => {
    let scale = 1;
    const frames: number[] = [];
    for (let i = 0; i < 60 && scale !== 2; i++) {
      scale = nextZoomFrameScale(scale, 2, 16.7);
      frames.push(scale);
    }
    expect(frames.length).toBeGreaterThan(3);
    expect(frames.length).toBeLessThan(40);
    expect(frames.every((s, i) => i === 0 || s > frames[i - 1])).toBe(true);
    expect(scale).toBe(2);
    // A frame of no time makes no progress; the same call with a target
    // already reached returns it unchanged.
    expect(nextZoomFrameScale(1, 2, 0)).toBe(1);
    expect(nextZoomFrameScale(2, 2, 16.7)).toBe(2);
  });

  it("keeps the image point under the cursor fixed while zooming", () => {
    const position = { x: -300, y: -120 };
    const cursor = { x: 500, y: 400 };
    const scale = 1.5;
    const nextScale = 2.4;
    const imagePoint = {
      x: (cursor.x - position.x) / scale,
      y: (cursor.y - position.y) / scale,
    };
    const next = zoomAnchoredPosition(position, scale, nextScale, cursor);
    expect(next.x + imagePoint.x * nextScale).toBeCloseTo(cursor.x, 9);
    expect(next.y + imagePoint.y * nextScale).toBeCloseTo(cursor.y, 9);
  });
});
