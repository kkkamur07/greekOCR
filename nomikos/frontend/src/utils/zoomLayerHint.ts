import { useCallback, useEffect, useRef, useState } from "react";

/**
 * How long the view must stay still before the compositor hint is dropped.
 * Long enough to bridge the gaps between frames of a wheel zoom or a momentum
 * pan, short enough that the re-raster reads as instant.
 */
export const ZOOM_SETTLE_MS = 180;

/**
 * Tracks whether a pan/zoom surface is currently moving.
 *
 * `will-change: transform` is worth having while the view moves: the pan/zoom
 * library rewrites the transform every frame, and the promoted layer keeps a
 * full-resolution scan and its overlay off the main thread. Leaving the hint on
 * permanently is not, because Chromium then pins the layer's raster scale: once
 * the user zooms in, the SVG overlay is no longer re-rastered, it is stretched
 * from the bitmap captured at the old scale, so vector geometry blurs exactly
 * like the scan underneath it.
 *
 * So the hint is applied while the transform is changing and removed once it
 * settles, which is the point at which the browser re-rasters at the new scale.
 * The timer is restarted by every transform, so no start/stop event pairing is
 * needed and the hint cannot get stuck on: any path through the library
 * (wheel, pinch, drag, momentum, animated zoom buttons, an imperative
 * setTransform) ends in a last transform followed by silence.
 */
export function useZoomLayerHint(settleMs: number = ZOOM_SETTLE_MS) {
  const [transforming, setTransforming] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const markTransforming = useCallback(() => {
    setTransforming(true);
    if (timerRef.current !== null) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      setTransforming(false);
    }, settleMs);
  }, [settleMs]);

  useEffect(
    () => () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    },
    [],
  );

  return { transforming, markTransforming };
}
