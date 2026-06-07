import { describe, expect, it } from "vitest";

import { fitTransform } from "./usePanZoom";

describe("fitTransform", () => {
  it("refits the manuscript when the canvas container shrinks for a side-by-side PDF panel", () => {
    const imageWidth = 2000;
    const imageHeight = 3000;
    const fullWidth = fitTransform(800, 600, imageWidth, imageHeight);
    const halfWidth = fitTransform(400, 600, imageWidth, imageHeight);

    expect(halfWidth.scale).toBeLessThan(fullWidth.scale);
    expect(halfWidth.x).toBeGreaterThan(0);
    expect(halfWidth.y).toBeGreaterThan(0);
  });
});
