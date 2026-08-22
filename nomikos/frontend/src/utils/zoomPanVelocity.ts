import type { ReactZoomPanPinchRef } from "react-zoom-pan-pinch";

/**
 * Clear react-zoom-pan-pinch's pan-velocity tracking at the start of a pan.
 *
 * The library (3.7) measures velocity from the change in the transform position
 * between consecutive move events, but it only initialises `lastMousePosition`
 * and `velocityTime` in the constructor and never clears them when a gesture
 * ends. Anything that moves the content without a pan (a wheel zoom, the zoom
 * buttons, "fit to view", a velocity animation) leaves a stale reference point
 * behind, and the first move of the next gesture is measured against it. A
 * plain click with one pixel of jitter then looks like a swipe of several
 * thousand pixels, and `handlePanningEnd` flings the page off the screen.
 *
 * Pass this as `onPanningStart` so every gesture measures only its own motion.
 */
export function resetPanVelocityTracking(
  ref: Pick<ReactZoomPanPinchRef, "instance">,
): void {
  const { instance } = ref;
  instance.velocity = null;
  instance.velocityTime = null;
  instance.lastMousePosition = null;
}
