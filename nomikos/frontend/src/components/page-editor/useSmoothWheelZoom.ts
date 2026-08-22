import { useEffect, type RefObject } from "react";
import type { ReactZoomPanPinchRef } from "react-zoom-pan-pinch";

import {
  PINCH_ZOOM_RATE,
  WHEEL_ZOOM_RATE,
  nextZoomFrameScale,
  wheelDeltaPixels,
  wheelTargetScale,
  zoomAnchoredPosition,
} from "./canvasZoom";

type SmoothWheelZoomOptions = {
  minScale: number;
  maxScale: number;
  /** Multiplier on the wheel rate; a trackpad pinch (ctrl+wheel) ignores it. */
  speed: number;
};

/**
 * Replace react-zoom-pan-pinch's wheel zoom with a cursor-anchored glide.
 *
 * Wheel events only move a target scale (multiplicatively, see canvasZoom.ts);
 * a requestAnimationFrame loop then eases the real scale toward it, keeping
 * the image point under the cursor fixed, and writes each frame through the
 * library's `setTransform` so its state, bounds and `onTransformed` stay in
 * step. Mount with `wheel={{ disabled: true }}` on the TransformWrapper.
 */
export function useSmoothWheelZoom(
  hostRef: RefObject<HTMLElement | null>,
  transformRef: RefObject<ReactZoomPanPinchRef | null>,
  { minScale, maxScale, speed }: SmoothWheelZoomOptions,
): void {
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let target: number | null = null;
    let cursor = { x: 0, y: 0 };
    let frame: number | null = null;
    let lastFrameTime = 0;

    const tick = (now: number) => {
      frame = null;
      const ref = transformRef.current;
      if (!ref || target === null) return;
      const { instance } = ref;
      // A button, double-click or fit animation took over: let it finish.
      if (instance.animate) {
        target = null;
        return;
      }
      const { scale, positionX, positionY } = instance.transformState;
      const dt = lastFrameTime ? now - lastFrameTime : 16;
      lastFrameTime = now;
      const next = nextZoomFrameScale(scale, target, dt);
      const position = zoomAnchoredPosition(
        { x: positionX, y: positionY },
        scale,
        next,
        cursor,
      );
      ref.setTransform(position.x, position.y, next, 0);
      if (next === target) {
        target = null;
        return;
      }
      frame = requestAnimationFrame(tick);
    };

    const onWheel = (event: WheelEvent) => {
      const ref = transformRef.current;
      if (!ref) return;
      const { instance } = ref;
      const wrapper = instance.wrapperComponent;
      if (!wrapper || instance.isPanning) return;
      if (!(event.target instanceof Node) || !wrapper.contains(event.target)) {
        return;
      }
      event.preventDefault();

      // Take over from any running library animation so the two do not fight.
      if (typeof instance.animation === "number") {
        cancelAnimationFrame(instance.animation);
      }
      instance.animation = null;
      instance.animate = false;

      const rect = wrapper.getBoundingClientRect();
      cursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      const rate = event.ctrlKey ? PINCH_ZOOM_RATE : WHEEL_ZOOM_RATE * speed;
      target = wheelTargetScale(
        target ?? instance.transformState.scale,
        wheelDeltaPixels(event.deltaY, event.deltaMode),
        rate,
        minScale,
        maxScale,
      );
      if (frame === null) {
        lastFrameTime = 0;
        frame = requestAnimationFrame(tick);
      }
    };

    host.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      host.removeEventListener("wheel", onWheel);
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [hostRef, transformRef, minScale, maxScale, speed]);
}
