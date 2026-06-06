"use client";

import { useCallback, useRef, useState } from "react";

export interface PanZoomState {
  x: number;
  y: number;
  scale: number;
}

const MIN_SCALE = 0.05;
const MAX_SCALE = 12;

function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}

export function centerAtFullScale(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
): PanZoomState {
  if (containerWidth <= 0 || containerHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return { x: 0, y: 0, scale: 1 };
  }
  return {
    scale: 1,
    x: (containerWidth - imageWidth) / 2,
    y: (containerHeight - imageHeight) / 2,
  };
}

export function fitTransform(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
): PanZoomState {
  if (containerWidth <= 0 || containerHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return { x: 0, y: 0, scale: 1 };
  }
  const padding = 16;
  const scale = Math.min(
    (containerWidth - padding) / imageWidth,
    (containerHeight - padding) / imageHeight,
  );
  const safeScale = clampScale(scale);
  return {
    scale: safeScale,
    x: (containerWidth - imageWidth * safeScale) / 2,
    y: (containerHeight - imageHeight * safeScale) / 2,
  };
}

export function imageCoordsFromTransform(
  container: HTMLElement,
  clientX: number,
  clientY: number,
  transform: PanZoomState,
): [number, number] {
  const rect = container.getBoundingClientRect();
  const x = (clientX - rect.left - transform.x) / transform.scale;
  const y = (clientY - rect.top - transform.y) / transform.scale;
  return [Math.round(x), Math.round(y)];
}

export function usePanZoom(imageWidth: number, imageHeight: number) {
  const [transform, setTransform] = useState<PanZoomState>({ x: 0, y: 0, scale: 1 });
  const transformRef = useRef(transform);
  transformRef.current = transform;
  const containerRef = useRef<HTMLDivElement>(null);
  const panSession = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(
    null,
  );

  const fitPage = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    setTransform(fitTransform(el.clientWidth, el.clientHeight, imageWidth, imageHeight));
  }, [imageWidth, imageHeight]);

  const centerPage = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    setTransform(centerAtFullScale(el.clientWidth, el.clientHeight, imageWidth, imageHeight));
  }, [imageWidth, imageHeight]);

  const zoomBy = useCallback(
    (factor: number, focalX?: number, focalY?: number) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const mx = focalX ?? rect.width / 2;
      const my = focalY ?? rect.height / 2;

      setTransform((prev) => {
        const nextScale = clampScale(prev.scale * factor);
        const ratio = nextScale / prev.scale;
        return {
          scale: nextScale,
          x: mx - (mx - prev.x) * ratio,
          y: my - (my - prev.y) * ratio,
        };
      });
    },
    [],
  );

  const zoomIn = useCallback(() => zoomBy(1.2), [zoomBy]);
  const zoomOut = useCallback(() => zoomBy(1 / 1.2), [zoomBy]);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      zoomBy(factor, mx, my);
    },
    [zoomBy],
  );

  const startPan = useCallback((clientX: number, clientY: number) => {
    const t = transformRef.current;
    panSession.current = {
      startX: clientX,
      startY: clientY,
      originX: t.x,
      originY: t.y,
    };
  }, []);

  const movePan = useCallback((clientX: number, clientY: number) => {
    const session = panSession.current;
    if (!session) return false;
    setTransform((prev) => ({
      ...prev,
      x: session.originX + (clientX - session.startX),
      y: session.originY + (clientY - session.startY),
    }));
    return (
      Math.abs(clientX - session.startX) > 2 || Math.abs(clientY - session.startY) > 2
    );
  }, []);

  const endPan = useCallback(() => {
    panSession.current = null;
  }, []);

  return {
    containerRef,
    transform,
    setTransform,
    fitPage,
    centerPage,
    zoomIn,
    zoomOut,
    handleWheel,
    startPan,
    movePan,
    endPan,
  };
}
