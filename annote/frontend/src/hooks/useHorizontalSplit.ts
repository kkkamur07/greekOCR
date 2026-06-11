"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseHorizontalSplitOptions {
  enabled: boolean;
  minLeadingPx?: number;
  minTrailingPx?: number;
  defaultTrailingRatio?: number;
  dividerPx?: number;
}

export function useHorizontalSplit(
  containerRef: React.RefObject<HTMLElement | null>,
  {
    enabled,
    minLeadingPx = 280,
    minTrailingPx = 240,
    defaultTrailingRatio = 0.5,
    dividerPx = 6,
  }: UseHorizontalSplitOptions,
) {
  const [trailingWidth, setTrailingWidth] = useState<number | null>(null);
  const draggingRef = useRef(false);

  const clampTrailing = useCallback(
    (width: number, containerWidth: number) => {
      const maxTrailing = containerWidth - minLeadingPx - dividerPx;
      return Math.max(minTrailingPx, Math.min(maxTrailing, width));
    },
    [dividerPx, minLeadingPx, minTrailingPx],
  );

  useEffect(() => {
    if (!enabled) return;
    const el = containerRef.current;
    if (!el) return;

    const init = () => {
      setTrailingWidth((prev) => {
        if (prev != null) return clampTrailing(prev, el.clientWidth);
        return clampTrailing(el.clientWidth * defaultTrailingRatio, el.clientWidth);
      });
    };

    init();

    const observer = new ResizeObserver(() => {
      setTrailingWidth((prev) => {
        if (prev == null) return prev;
        return clampTrailing(prev, el.clientWidth);
      });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [enabled, containerRef, clampTrailing, defaultTrailingRatio]);

  const onDividerPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    draggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const onDividerPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const next = clampTrailing(rect.right - e.clientX, rect.width);
      setTrailingWidth(next);
    },
    [clampTrailing, containerRef],
  );

  const endDrag = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  const resetTrailing = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    setTrailingWidth(clampTrailing(el.clientWidth * defaultTrailingRatio, el.clientWidth));
  }, [clampTrailing, containerRef, defaultTrailingRatio]);

  return {
    trailingWidth,
    dividerProps: {
      onPointerDown: onDividerPointerDown,
      onPointerMove: onDividerPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      onDoubleClick: resetTrailing,
    },
  };
}
