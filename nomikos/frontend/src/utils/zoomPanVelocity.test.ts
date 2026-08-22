import { describe, expect, it } from "vitest";

import { resetPanVelocityTracking } from "./zoomPanVelocity";

describe("resetPanVelocityTracking", () => {
  it("drops the reference point left behind by the previous gesture", () => {
    // What the library holds after a pan, a wheel zoom, and a pause: the last
    // pan's end position and time, which the next click would be measured against.
    const instance = {
      velocity: { velocityX: 4, velocityY: 4, total: 0.14 },
      velocityTime: 1_787_413_572_555,
      lastMousePosition: { x: 16, y: 16 },
    };

    resetPanVelocityTracking({ instance } as never);

    expect(instance).toEqual({
      velocity: null,
      velocityTime: null,
      lastMousePosition: null,
    });
  });
});
