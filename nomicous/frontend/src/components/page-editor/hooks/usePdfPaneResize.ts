import { useCallback, useEffect, useRef, useState } from "react";

const MIN_WIDTH = 180;
const DEFAULT_WIDTH = 280;
const MAX_VW = 0.5;

export function usePdfPaneResize(initialWidth = DEFAULT_WIDTH) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(initialWidth);
  const draggingRef = useRef(false);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      draggingRef.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      event.currentTarget.classList.add("is-dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      const parent = wrapRef.current?.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      const maxWidth = window.innerWidth * MAX_VW;
      const nextWidth = rect.right - event.clientX;
      setWidth(Math.min(maxWidth, Math.max(MIN_WIDTH, nextWidth)));
    },
    [],
  );

  const endDrag = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    event.currentTarget.classList.remove("is-dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    const onResize = () => {
      const maxWidth = window.innerWidth * MAX_VW;
      setWidth((current) => Math.min(current, maxWidth));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return {
    wrapRef,
    width,
    setWidth,
    onPointerDown,
    onPointerMove,
    onPointerUp: endDrag,
    onPointerCancel: endDrag,
  };
}
