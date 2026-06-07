"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { DrawTool, Segment, SegmentKind } from "@/types/api";
import SegmentOverlay, { MIN_SEGMENT_POINTS } from "./SegmentOverlay";
import ZoomControls from "./ZoomControls";
import { imageCoordsFromTransform, usePanZoom } from "./usePanZoom";
import "./ImageCanvas.css";

export interface ImageCanvasHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  fitPage: () => void;
  /** Cancel in-progress draw; returns true if something was cancelled. */
  cancelDraft: () => boolean;
}

interface ImageCanvasProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  segments: Segment[];
  selectedId: string | null;
  tool: DrawTool;
  editMode: boolean;
  readOnly?: boolean;
  showSegments: boolean;
  selectedVertexIndex: number | null;
  onSelect: (id: string | null) => void;
  onSelectVertex: (index: number | null) => void;
  onAddSegment: (segment: Segment) => void;
  onUpdateSegment: (segment: Segment) => void;
}

function nextSegmentNumber(segments: Segment[]): number {
  if (segments.length === 0) return 1;
  return Math.max(...segments.map((s) => s.number)) + 1;
}

function newId(): string {
  return `seg-${crypto.randomUUID().slice(0, 8)}`;
}

function rectFromDrag(start: [number, number], end: [number, number]): [number, number][] {
  const x1 = Math.min(start[0], end[0]);
  const y1 = Math.min(start[1], end[1]);
  const x2 = Math.max(start[0], end[0]);
  const y2 = Math.max(start[1], end[1]);
  return [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ];
}

const ImageCanvas = forwardRef<ImageCanvasHandle, ImageCanvasProps>(function ImageCanvas(
  {
    imageUrl,
    imageWidth,
    imageHeight,
    segments,
    selectedId,
    tool,
    editMode,
    readOnly = false,
    showSegments,
    selectedVertexIndex,
    onSelect,
    onSelectVertex,
    onAddSegment,
    onUpdateSegment,
  },
  ref,
) {
  const [draftPoints, setDraftPoints] = useState<[number, number][]>([]);
  const [rectPreview, setRectPreview] = useState<[number, number][]>([]);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [isPanning, setIsPanning] = useState(false);

  const panMovedRef = useRef(false);
  const suppressClickRef = useRef(false);
  const rectStartRef = useRef<[number, number] | null>(null);
  const drawingRectRef = useRef(false);
  const polygonPointerRef = useRef<{ x: number; y: number } | null>(null);
  const polygonDragRef = useRef(false);
  const draftPointsRef = useRef(draftPoints);
  draftPointsRef.current = draftPoints;

  const DRAG_THRESHOLD = 5;

  const cancelDraft = useCallback(() => {
    if (drawingRectRef.current || rectStartRef.current) {
      setRectPreview([]);
      rectStartRef.current = null;
      drawingRectRef.current = false;
      return true;
    }
    if (draftPointsRef.current.length > 0) {
      setDraftPoints((prev) => prev.slice(0, -1));
      return true;
    }
    return false;
  }, []);

  const {
    containerRef,
    transform,
    fitPage,
    centerPage,
    zoomIn,
    zoomOut,
    handleWheel,
    startPan,
    movePan,
    endPan,
  } = usePanZoom(imageWidth, imageHeight);

  const centeredRef = useRef(false);

  useImperativeHandle(ref, () => ({ zoomIn, zoomOut, fitPage, cancelDraft }), [
    zoomIn,
    zoomOut,
    fitPage,
    cancelDraft,
  ]);

  useEffect(() => {
    centeredRef.current = false;
  }, [imageWidth, imageHeight]);

  useEffect(() => {
    if (centeredRef.current) return;
    const el = containerRef.current;
    if (!el || imageWidth <= 0 || imageHeight <= 0) return;
    centerPage();
    centeredRef.current = true;
  }, [centerPage, containerRef, imageWidth, imageHeight]);

  const imageCoords = useCallback(
    (clientX: number, clientY: number) => {
      const container = containerRef.current;
      if (!container) return null;
      return imageCoordsFromTransform(container, clientX, clientY, transform);
    },
    [containerRef, transform],
  );

  const finishSegment = useCallback(
    (kind: SegmentKind, points: [number, number][]) => {
      if (readOnly || points.length < 4) return;
      suppressClickRef.current = true;
      window.setTimeout(() => {
        suppressClickRef.current = false;
      }, 0);
      onAddSegment({
        id: newId(),
        number: nextSegmentNumber(segments),
        kind,
        points,
        paired_text_line_index: null,
      });
      setDraftPoints([]);
      setRectPreview([]);
      rectStartRef.current = null;
      drawingRectRef.current = false;
    },
    [onAddSegment, readOnly, segments],
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;

      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        setSpaceHeld(true);
        return;
      }
      if (e.key === "Enter" && tool === "polygon" && draftPoints.length >= 4) {
        e.preventDefault();
        finishSegment("polygon", draftPoints);
        return;
      }
      if (e.key === "Escape") {
        setDraftPoints([]);
        setRectPreview([]);
        rectStartRef.current = null;
        drawingRectRef.current = false;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceHeld(false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [tool, draftPoints, finishSegment]);

  const isDrawingTool = tool === "polygon" || tool === "rectangle";

  const isSegmentHit = (target: EventTarget | null) =>
    target instanceof Element &&
    (target.closest("[data-segment-hit]") != null || target.closest("[data-vertex-handle]") != null);

  const canPanWithPointer = useCallback(
    (button: number) => {
      if (button === 1 || button === 2) return true;
      if (button !== 0) return false;
      if (spaceHeld || tool === "pan" || tool === "select") return true;
      if (isDrawingTool && spaceHeld) return true;
      return false;
    },
    [spaceHeld, tool, isDrawingTool],
  );

  const handleStagePointerDownCapture = (e: React.PointerEvent) => {
    if (isSegmentHit(e.target)) {
      panMovedRef.current = false;
      return;
    }

    if (readOnly) {
      if (!canPanWithPointer(e.button)) return;
      panMovedRef.current = false;
      setIsPanning(true);
      startPan(e.clientX, e.clientY);
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }

    if (tool === "rectangle" && !editMode && !spaceHeld && e.button === 0) {
      const pt = imageCoords(e.clientX, e.clientY);
      if (!pt) return;
      rectStartRef.current = pt;
      drawingRectRef.current = true;
      setRectPreview(rectFromDrag(pt, pt));
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }

    if (tool === "polygon" && !editMode && !spaceHeld && e.button === 0) {
      polygonPointerRef.current = { x: e.clientX, y: e.clientY };
      polygonDragRef.current = false;
      panMovedRef.current = false;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }

    if (!canPanWithPointer(e.button)) return;
    if (tool === "rectangle" && !spaceHeld && e.button === 0) return;

    panMovedRef.current = false;
    setIsPanning(true);
    startPan(e.clientX, e.clientY);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const handleStagePointerMove = (e: React.PointerEvent) => {
    if (tool === "rectangle" && drawingRectRef.current && rectStartRef.current) {
      const pt = imageCoords(e.clientX, e.clientY);
      if (!pt) return;
      setRectPreview(rectFromDrag(rectStartRef.current, pt));
      return;
    }

    if (tool === "polygon" && polygonPointerRef.current && !spaceHeld) {
      const start = polygonPointerRef.current;
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      if (!polygonDragRef.current && Math.hypot(dx, dy) > DRAG_THRESHOLD) {
        polygonDragRef.current = true;
        setIsPanning(true);
        startPan(start.x, start.y);
        movePan(e.clientX, e.clientY);
      }
    }

    if (movePan(e.clientX, e.clientY)) {
      panMovedRef.current = true;
    }
  };

  const handleStagePointerUp = (e: React.PointerEvent) => {
    if (tool === "rectangle" && drawingRectRef.current && rectStartRef.current) {
      const pt = imageCoords(e.clientX, e.clientY);
      if (pt) {
        const points = rectFromDrag(rectStartRef.current, pt);
        const w = Math.abs(points[2][0] - points[0][0]);
        const h = Math.abs(points[2][1] - points[0][1]);
        if (w > 8 && h > 8) {
          finishSegment("rectangle", points);
        } else {
          setRectPreview([]);
          rectStartRef.current = null;
          drawingRectRef.current = false;
        }
      }
    }

    if (tool === "polygon" && polygonPointerRef.current && !spaceHeld && !polygonDragRef.current) {
      const pt = imageCoords(e.clientX, e.clientY);
      if (pt) {
        setDraftPoints((prev) => [...prev, pt]);
      }
    }

    polygonPointerRef.current = null;
    polygonDragRef.current = false;
    setIsPanning(false);
    endPan();
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* capture may already be released */
    }
  };

  const handleLayerClick = (e: React.MouseEvent) => {
    if (
      suppressClickRef.current ||
      panMovedRef.current ||
      spaceHeld ||
      tool === "pan" ||
      isSegmentHit(e.target)
    ) {
      return;
    }

    if (editMode && !isSegmentHit(e.target)) {
      onSelectVertex(null);
      return;
    }

    if (tool === "select" && !editMode) {
      onSelect(null);
    }
  };

  const handleLayerDoubleClick = () => {
    if (panMovedRef.current) return;
    if (tool === "polygon" && draftPoints.length >= 4) {
      finishSegment("polygon", draftPoints);
    }
  };

  const overlayDraft = rectPreview.length > 0 ? rectPreview : draftPoints;
  const displayWidth = imageWidth * transform.scale;
  const displayHeight = imageHeight * transform.scale;
  const stageClass = [
    "image-canvas-stage",
    isPanning ? "is-panning" : "",
    tool === "rectangle" && !spaceHeld ? "is-drawing" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="image-canvas-root">
      <div
        ref={containerRef}
        className={stageClass}
        onWheel={handleWheel}
        onPointerDownCapture={handleStagePointerDownCapture}
        onPointerMove={handleStagePointerMove}
        onPointerUp={handleStagePointerUp}
        onPointerCancel={handleStagePointerUp}
      >
        <div
          className="image-canvas-layer"
          style={{
            width: displayWidth,
            height: displayHeight,
            transform: `translate(${transform.x}px, ${transform.y}px)`,
          }}
          onClick={handleLayerClick}
          onDoubleClick={handleLayerDoubleClick}
        >
          <div className="image-page-layer">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl}
              alt="Manuscript page"
              width={imageWidth}
              height={imageHeight}
              draggable={false}
            />
            <SegmentOverlay
              imageWidth={imageWidth}
              imageHeight={imageHeight}
              segments={segments}
              selectedId={selectedId}
              draftPoints={overlayDraft}
              editMode={editMode}
              visible={showSegments}
              zoomScale={transform.scale}
              interactive={!readOnly && !editMode && !(tool === "polygon" && !spaceHeld)}
              onSelect={(id) => {
                if (!panMovedRef.current) onSelect(id);
              }}
              clientToImage={imageCoords}
              selectedVertexIndex={selectedVertexIndex}
              onSelectVertex={onSelectVertex}
              onVertexDrag={(id, idx, x, y) => {
                if (readOnly) return;
                const seg = segments.find((s) => s.id === id);
                if (!seg) return;
                const points = seg.points.map((p, i) =>
                  i === idx ? ([x, y] as [number, number]) : p,
                );
                onUpdateSegment({ ...seg, points });
              }}
              onInsertVertex={(id, afterIndex, x, y) => {
                if (readOnly) return;
                const seg = segments.find((s) => s.id === id);
                if (!seg) return;
                const points = [...seg.points];
                points.splice(afterIndex + 1, 0, [x, y]);
                onUpdateSegment({ ...seg, points });
              }}
              onRemoveVertex={(id, pointIndex) => {
                if (readOnly) return;
                const seg = segments.find((s) => s.id === id);
                if (!seg || seg.points.length <= MIN_SEGMENT_POINTS) return;
                const points = seg.points.filter((_, i) => i !== pointIndex);
                onUpdateSegment({ ...seg, points });
                if (selectedVertexIndex === pointIndex) {
                  onSelectVertex(null);
                } else if (selectedVertexIndex != null && selectedVertexIndex > pointIndex) {
                  onSelectVertex(selectedVertexIndex - 1);
                }
              }}
            />
          </div>
        </div>
      </div>

      <div className="image-canvas-viewport flex items-start justify-end p-3">
        <ZoomControls zoomLevel={transform.scale} onZoomIn={zoomIn} onZoomOut={zoomOut} onFitPage={fitPage} />
      </div>
    </div>
  );
});

export default ImageCanvas;
