import { useEffect, useRef, useState, type MouseEvent } from 'react';
import {
  TransformComponent,
  TransformWrapper,
  type ReactZoomPanPinchRef,
} from 'react-zoom-pan-pinch';
import type { LinePoint, LineResponse, PartLayoutResponse } from '../../api/client';
import { AuthenticatedImage } from '../AuthenticatedImage';
import { canvasHandleRadius, canvasStrokeWidth, insertPolygonVertexAtClick, normalizeGeometryPoints, points, rectanglePoints, removePolygonVertex } from './canvasGeometry';
import type { PageEditorCanvasSettings } from './pageEditorSettings';

const ZOOM_ANIMATION_MS = 220;
const ZOOM_BUTTON_STEP = 0.12;

type CanvasSurfaceProps = {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  selectedSegmentId: string | null;
  pairedSegmentIds: Set<string>;
  drawingRectangle: boolean;
  drawingPolygon: boolean;
  draftStart: LinePoint | null;
  draftEnd: LinePoint | null;
  draftPolygon: LinePoint[];
  draftPolygonCursor: LinePoint | null;
  onDraftPolygonCursor: (point: LinePoint | null) => void;
  settings: PageEditorCanvasSettings;
  zoomLevel: number;
  onDraftStart: (point: LinePoint) => void;
  onDraftMove: (point: LinePoint) => void;
  onRectangleDrawn: (point: LinePoint) => void;
  onPolygonPoint: (point: LinePoint) => void;
  onPolygonComplete: () => void;
  onSelectLine: (lineId: string) => void;
  onSelectSegment: (lineId: string) => void;
  segmentVertexEditEnabled: boolean;
  suppressBaselineSegmentId: string | null;
  vertexEditPoints: LinePoint[] | null;
  draggedVertexIndex: number | null;
  pendingVertexIndex: number | null;
  onVertexPointerDown: (vertexIndex: number) => void;
  onVertexPointerMove: (point: LinePoint) => void;
  onInsertVertexOnEdge: (nextPoints: LinePoint[]) => void;
};

const VERTEX_DRAG_THRESHOLD = 3;

function CanvasSurface({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  selectedSegmentId,
  pairedSegmentIds,
  drawingRectangle,
  drawingPolygon,
  draftStart,
  draftEnd,
  draftPolygon,
  draftPolygonCursor,
  onDraftPolygonCursor,
  settings,
  zoomLevel,
  onDraftStart,
  onDraftMove,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
  segmentVertexEditEnabled,
  suppressBaselineSegmentId,
  vertexEditPoints,
  draggedVertexIndex,
  pendingVertexIndex,
  onVertexPointerDown,
  onVertexPointerMove,
  onInsertVertexOnEdge,
}: CanvasSurfaceProps) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const canvasWidth = naturalSize?.width ?? imageWidth;
  const canvasHeight = naturalSize?.height ?? imageHeight;
  const canvasMax = Math.max(canvasWidth, canvasHeight);

  const strokeWidth = (base: number) =>
    canvasStrokeWidth(base, zoomLevel, settings.overlayStrokeWidth, canvasMax);
  const baselineStroke = (manual: boolean) =>
    canvasStrokeWidth(
      (manual ? 1.0 : 1.2) * settings.baselineStrokeWidth,
      zoomLevel,
      settings.overlayStrokeWidth,
      canvasMax,
    );
  const handleRadius = (base: number) =>
    canvasHandleRadius(
      base,
      zoomLevel,
      settings.handleSize,
      settings.overlayStrokeWidth,
      canvasMax,
    );
  const segmentFill = (r: number, g: number, b: number) =>
    `rgba(${r}, ${g}, ${b}, ${settings.segmentFillOpacity})`;

  const eventPoint = (event: MouseEvent<SVGElement>): LinePoint => {
    const svg = event.currentTarget.ownerSVGElement ?? event.currentTarget;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return [event.clientX, event.clientY];
    }
    return [
      Math.round(((event.clientX - rect.left) / rect.width) * canvasWidth),
      Math.round(((event.clientY - rect.top) / rect.height) * canvasHeight),
    ];
  };

  const edgeHitDistance = Math.max(handleRadius(3.5), strokeWidth(10));

  const draftRectangle =
    drawingRectangle && draftStart && draftEnd ? rectanglePoints(draftStart, draftEnd) : null;
  const isDraggingVertex = draggedVertexIndex !== null;
  const isVertexInteracting = pendingVertexIndex !== null || isDraggingVertex;

  return (
    <div
      style={{
        position: 'relative',
        width: canvasWidth,
        height: canvasHeight,
        background: '#fff',
        boxShadow: '0 8px 40px rgba(0, 0, 0, 0.5)',
      }}
    >
      <AuthenticatedImage
        compact
        src={imageUrl}
        alt={imageAlt}
        onLoad={(event) => {
          const { naturalWidth, naturalHeight } = event.currentTarget;
          if (naturalWidth > 0 && naturalHeight > 0) {
            setNaturalSize({ width: naturalWidth, height: naturalHeight });
          }
        }}
        style={{
          display: 'block',
          width: canvasWidth,
          height: canvasHeight,
          userSelect: 'none',
          pointerEvents: 'none',
        }}
      />
      <svg
        role="img"
        aria-label="Page geometry canvas"
        viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
        onMouseDown={(event) => {
          if (!drawingRectangle) return;
          event.stopPropagation();
          onDraftStart(eventPoint(event));
        }}
        onMouseMove={(event) => {
          if (drawingPolygon) {
            onDraftPolygonCursor(eventPoint(event));
          }
          if (isVertexInteracting) {
            event.stopPropagation();
            onVertexPointerMove(eventPoint(event));
            return;
          }
          if (!drawingRectangle || !draftStart) return;
          event.stopPropagation();
          onDraftMove(eventPoint(event));
        }}
        onMouseUp={(event) => {
          if (!drawingRectangle || !draftStart) return;
          event.stopPropagation();
          onRectangleDrawn(eventPoint(event));
        }}
        onClick={(event) => {
          if (!drawingPolygon) return;
          event.stopPropagation();
          onPolygonPoint(eventPoint(event));
        }}
        onDoubleClick={(event) => {
          if (!drawingPolygon) return;
          event.stopPropagation();
          onPolygonComplete();
        }}
        style={{
          position: 'absolute',
          inset: 0,
          width: canvasWidth,
          height: canvasHeight,
          cursor:
            drawingRectangle || drawingPolygon
              ? 'crosshair'
              : isVertexInteracting
                ? 'grabbing'
                : segmentVertexEditEnabled
                  ? 'default'
                  : 'grab',
          touchAction: 'none',
        }}
      >
        {settings.showLayoutBlocks &&
          layout.blocks.map((block) => (
          <polygon
            key={block.id}
            aria-label={`Block ${block.id}`}
            points={points(block.box)}
            fill="rgba(216, 199, 161, 0.08)"
            stroke="#d8c7a1"
            strokeWidth={strokeWidth(1.2)}
          />
        ))}
        {lines
          .slice()
          .sort((a, b) => a.order - b.order)
          .map((line) => {
            const selected = line.id === selectedSegmentId;
            const paired = pairedSegmentIds.has(line.id);
            const fill = selected
              ? segmentFill(180, 0, 0)
              : paired
                ? segmentFill(4, 120, 87)
                : segmentFill(180, 83, 9);
            const strokeColor = selected
              ? 'var(--red, #b40000)'
              : paired
                ? 'rgba(4, 120, 87, 0.75)'
                : 'rgba(180, 83, 9, 0.55)';
            const segmentPoints =
              selected && vertexEditPoints && vertexEditPoints.length >= 3
                ? vertexEditPoints
                : line.points;
            return (
              <polygon
                key={line.id}
                aria-label={`Segment ${line.order + 1}${paired ? ', paired' : ''}`}
                aria-current={selected ? 'true' : undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  if (selected && segmentVertexEditEnabled && segmentPoints.length >= 3) {
                    const click = eventPoint(event);
                    const nextPoints = insertPolygonVertexAtClick(
                      segmentPoints,
                      click,
                      edgeHitDistance,
                    );
                    if (nextPoints) {
                      onInsertVertexOnEdge(nextPoints);
                      return;
                    }
                  }
                  onSelectSegment(line.id);
                }}
                points={points(segmentPoints)}
                fill={fill}
                stroke={strokeColor}
                strokeWidth={strokeWidth(selected ? 2.2 : paired ? 1.8 : 1.6)}
                style={
                  selected && segmentVertexEditEnabled
                    ? { pointerEvents: 'all', cursor: 'copy' }
                    : undefined
                }
              />
            );
          })}
        {settings.showBaselines &&
          lines
            .slice()
            .sort((a, b) => a.order - b.order)
            .map((line) => {
              if (line.id === suppressBaselineSegmentId) return null;
              if (normalizeGeometryPoints(line.baseline).length < 2) return null;
              return (
                <polyline
                  key={`baseline-${line.id}`}
                  aria-label={`Line ${line.id} baseline`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectLine(line.id);
                  }}
                  points={points(line.baseline)}
                  fill="none"
                  stroke={line.manual_geometry ? '#059669' : '#0d9488'}
                  strokeLinecap="round"
                  strokeWidth={baselineStroke(Boolean(line.manual_geometry))}
                  strokeDasharray={
                    line.manual_geometry
                      ? undefined
                      : `${baselineStroke(false) * 2.5},${baselineStroke(false) * 1.25}`
                  }
                  style={{ pointerEvents: 'stroke' }}
                />
              );
            })}
        {draftRectangle && (
          <polygon
            aria-label="Draft rectangle segment"
            points={points(draftRectangle)}
            fill="rgba(13, 31, 60, 0.12)"
            stroke="rgba(13, 31, 60, 0.75)"
            strokeWidth={strokeWidth(2)}
            strokeDasharray={`${strokeWidth(6)},${strokeWidth(3)}`}
          />
        )}
        {drawingPolygon && draftPolygon.length > 0 && (
          <>
            {draftPolygon.map(([x, y], index) => {
              if (index === 0) return null;
              const [prevX, prevY] = draftPolygon[index - 1];
              return (
                <line
                  key={`draft-segment-${index}`}
                  x1={prevX}
                  y1={prevY}
                  x2={x}
                  y2={y}
                  stroke="rgba(13, 31, 60, 0.85)"
                  strokeWidth={strokeWidth(2)}
                  strokeLinecap="round"
                />
              );
            })}
            {draftPolygonCursor && draftPolygon.length > 0 && (
              <>
                <line
                  aria-hidden="true"
                  x1={draftPolygon[draftPolygon.length - 1][0]}
                  y1={draftPolygon[draftPolygon.length - 1][1]}
                  x2={draftPolygonCursor[0]}
                  y2={draftPolygonCursor[1]}
                  stroke="rgba(13, 31, 60, 0.65)"
                  strokeWidth={strokeWidth(1.8)}
                  strokeDasharray={`${strokeWidth(6)},${strokeWidth(3)}`}
                  strokeLinecap="round"
                />
                {draftPolygon.length >= 2 && (
                  <line
                    aria-hidden="true"
                    x1={draftPolygonCursor[0]}
                    y1={draftPolygonCursor[1]}
                    x2={draftPolygon[0][0]}
                    y2={draftPolygon[0][1]}
                    stroke="rgba(4, 120, 87, 0.65)"
                    strokeWidth={strokeWidth(1.4)}
                    strokeDasharray={`${strokeWidth(5)},${strokeWidth(3)}`}
                    strokeLinecap="round"
                  />
                )}
              </>
            )}
            {draftPolygon.map(([x, y], index) => (
              <circle
                key={`draft-vertex-${index}`}
                cx={x}
                cy={y}
                r={handleRadius(2.2)}
                fill="#fff"
                stroke="rgba(13, 31, 60, 0.85)"
                strokeWidth={strokeWidth(0.8)}
              />
            ))}
          </>
        )}
        {segmentVertexEditEnabled &&
          selectedSegmentId &&
          vertexEditPoints &&
          vertexEditPoints.map(([x, y], index) => (
            <circle
              key={`segment-vertex-${selectedSegmentId}-${index}`}
              cx={x}
              cy={y}
              r={handleRadius(2.4)}
              fill="#fff"
              stroke="var(--red, #b40000)"
              strokeWidth={strokeWidth(0.9)}
              style={{
                cursor: isDraggingVertex ? 'grabbing' : 'pointer',
                pointerEvents: 'all',
              }}
              aria-label={`Segment vertex ${index + 1} · click to remove · drag to move`}
              onMouseDown={(event) => {
                event.stopPropagation();
                onVertexPointerDown(index);
              }}
            />
          ))}
      </svg>
    </div>
  );
}

type PageEditorCanvasProps = Omit<
  CanvasSurfaceProps,
  | 'draftEnd'
  | 'onDraftMove'
  | 'zoomLevel'
  | 'vertexEditPoints'
  | 'draggedVertexIndex'
  | 'pendingVertexIndex'
  | 'onVertexPointerDown'
  | 'onVertexPointerMove'
  | 'onInsertVertexOnEdge'
  | 'draftPolygonCursor'
  | 'onDraftPolygonCursor'
> & {
  onSegmentPointsChange: (segmentId: string, points: LinePoint[]) => void | Promise<void>;
};

export function PageEditorCanvas({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  selectedSegmentId,
  pairedSegmentIds,
  drawingRectangle,
  drawingPolygon,
  draftStart,
  draftPolygon,
  settings,
  segmentVertexEditEnabled,
  onDraftStart,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
  onSegmentPointsChange,
}: PageEditorCanvasProps) {
  const [zoomLevel, setZoomLevel] = useState(0.75);
  const [draftEnd, setDraftEnd] = useState<LinePoint | null>(null);
  const [draftPolygonCursor, setDraftPolygonCursor] = useState<LinePoint | null>(null);
  const [vertexEdit, setVertexEdit] = useState<{
    segmentId: string;
    points: LinePoint[];
    draggedIndex: number | null;
    pendingVertexIndex: number | null;
  } | null>(null);
  const transformRef = useRef<ReactZoomPanPinchRef>(null);
  const canvasWidth = imageWidth;
  const canvasHeight = imageHeight;
  const isDrawing = drawingRectangle || drawingPolygon;
  const isDraggingVertex = vertexEdit?.draggedIndex !== null && vertexEdit?.draggedIndex !== undefined;
  const isVertexInteracting =
    vertexEdit?.pendingVertexIndex !== null && vertexEdit?.pendingVertexIndex !== undefined
      ? true
      : isDraggingVertex;
  const selectedSegment = lines.find((line) => line.id === selectedSegmentId);
  const vertexEditPoints =
    vertexEdit?.segmentId === selectedSegmentId
      ? vertexEdit.points
      : selectedSegment?.points ?? null;

  useEffect(() => {
    if (!drawingRectangle) setDraftEnd(null);
  }, [drawingRectangle]);

  useEffect(() => {
    if (!drawingPolygon) setDraftPolygonCursor(null);
  }, [drawingPolygon]);

  useEffect(() => {
    if (!segmentVertexEditEnabled || !selectedSegmentId) {
      setVertexEdit(null);
    }
  }, [segmentVertexEditEnabled, selectedSegmentId]);

  useEffect(() => {
    if (!isVertexInteracting) return;
    const finishInteraction = () => {
      setVertexEdit((current) => {
        if (!current) return current;

        if (current.pendingVertexIndex !== null && current.draggedIndex === null) {
          const removed = removePolygonVertex(current.points, current.pendingVertexIndex);
          if (!removed) return null;
          const pending = {
            segmentId: current.segmentId,
            points: removed,
            draggedIndex: null as number | null,
            pendingVertexIndex: null as number | null,
          };
          void Promise.resolve(
            onSegmentPointsChange(pending.segmentId, pending.points),
          ).finally(() => {
            setVertexEdit((latest) =>
              latest?.segmentId === pending.segmentId ? null : latest,
            );
          });
          return pending;
        }

        if (current.draggedIndex === null) return current;
        const pending = {
          segmentId: current.segmentId,
          points: current.points,
          draggedIndex: null as number | null,
          pendingVertexIndex: null as number | null,
        };
        void Promise.resolve(
          onSegmentPointsChange(pending.segmentId, pending.points),
        ).finally(() => {
          setVertexEdit((latest) =>
            latest?.segmentId === pending.segmentId ? null : latest,
          );
        });
        return pending;
      });
    };
    window.addEventListener('mouseup', finishInteraction);
    return () => window.removeEventListener('mouseup', finishInteraction);
  }, [isVertexInteracting, onSegmentPointsChange]);

  const zoomAnimated = (direction: 'in' | 'out') => {
    const ref = transformRef.current;
    if (!ref) return;
    if (direction === 'in') ref.zoomIn(ZOOM_BUTTON_STEP, ZOOM_ANIMATION_MS);
    else ref.zoomOut(ZOOM_BUTTON_STEP, ZOOM_ANIMATION_MS);
  };

  return (
    <div className="pe-canvas-host">
      <TransformWrapper
        ref={transformRef}
        initialScale={0.75}
        minScale={0.15}
        maxScale={8}
        centerOnInit
        limitToBounds={false}
        wheel={{
          step: 0.06,
          smoothStep: 0.006,
          wheelDisabled: false,
          touchPadDisabled: false,
        }}
        pinch={{ step: 5, disabled: isDrawing }}
        doubleClick={{ disabled: isDrawing, step: 0.65, mode: 'zoomIn', animationTime: ZOOM_ANIMATION_MS }}
        panning={{
          disabled: isDrawing || isVertexInteracting,
          velocityDisabled: false,
          wheelPanning: false,
          allowLeftClickPan: true,
        }}
        zoomAnimation={{
          disabled: false,
          size: ZOOM_BUTTON_STEP,
          animationTime: ZOOM_ANIMATION_MS,
        }}
        velocityAnimation={{
          disabled: false,
          sensitivity: 1,
          animationTime: 350,
        }}
        alignmentAnimation={{ disabled: false, sizeX: 0, sizeY: 0, animationTime: ZOOM_ANIMATION_MS }}
        onTransformed={(ref) => setZoomLevel(ref.state.scale)}
      >
        {({ resetTransform, centerView }) => (
          <>
            <div className="pe-zoom">
              <button
                type="button"
                className="pe-zoom__btn"
                onClick={() => zoomAnimated('out')}
                aria-label="Zoom out"
              >
                −
              </button>
              <div className="pe-zoom__label" aria-live="polite">
                {Math.round(zoomLevel * 100)}%
              </div>
              <button
                type="button"
                className="pe-zoom__btn"
                onClick={() => zoomAnimated('in')}
                aria-label="Zoom in"
              >
                +
              </button>
              <button
                type="button"
                className="pe-zoom__btn"
                style={{ fontSize: '0.65rem' }}
                onClick={() => centerView(undefined, ZOOM_ANIMATION_MS)}
                aria-label="Fit page to view"
                title="Fit to view"
              >
                ⊡
              </button>
              <button
                type="button"
                className="pe-zoom__btn"
                style={{ fontSize: '0.65rem' }}
                onClick={() => resetTransform(undefined, ZOOM_ANIMATION_MS)}
                aria-label="Reset zoom"
                title="Reset zoom"
              >
                ⟲
              </button>
            </div>
            <TransformComponent
              wrapperStyle={{ width: '100%', height: '100%', touchAction: 'none' }}
              contentStyle={{ width: canvasWidth, height: canvasHeight }}
            >
              <CanvasSurface
                key={imageUrl}
                imageUrl={imageUrl}
                imageAlt={imageAlt}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                layout={layout}
                lines={lines}
                selectedSegmentId={selectedSegmentId}
                pairedSegmentIds={pairedSegmentIds}
                drawingRectangle={drawingRectangle}
                drawingPolygon={drawingPolygon}
                draftStart={draftStart}
                draftEnd={draftEnd}
                draftPolygon={draftPolygon}
                draftPolygonCursor={draftPolygonCursor}
                onDraftPolygonCursor={setDraftPolygonCursor}
                settings={settings}
                zoomLevel={zoomLevel}
                onDraftStart={(point) => {
                  setDraftEnd(point);
                  onDraftStart(point);
                }}
                onDraftMove={setDraftEnd}
                onRectangleDrawn={(point) => {
                  onRectangleDrawn(point);
                  setDraftEnd(null);
                }}
                onPolygonPoint={onPolygonPoint}
                onPolygonComplete={onPolygonComplete}
                onSelectLine={onSelectLine}
                onSelectSegment={onSelectSegment}
                segmentVertexEditEnabled={segmentVertexEditEnabled}
                suppressBaselineSegmentId={vertexEdit?.segmentId ?? null}
                vertexEditPoints={vertexEditPoints}
                draggedVertexIndex={vertexEdit?.draggedIndex ?? null}
                pendingVertexIndex={vertexEdit?.pendingVertexIndex ?? null}
                onVertexPointerDown={(vertexIndex) => {
                  if (!selectedSegmentId) return;
                  const basePoints =
                    vertexEdit?.segmentId === selectedSegmentId
                      ? vertexEdit.points
                      : selectedSegment?.points;
                  if (!basePoints || basePoints.length < 3) return;
                  setVertexEdit({
                    segmentId: selectedSegmentId,
                    points: [...basePoints],
                    draggedIndex: null,
                    pendingVertexIndex: vertexIndex,
                  });
                }}
                onVertexPointerMove={(point) => {
                  setVertexEdit((current) => {
                    if (!current) return current;
                    if (current.pendingVertexIndex !== null && current.draggedIndex === null) {
                      const anchor = current.points[current.pendingVertexIndex];
                      const moved =
                        Math.hypot(point[0] - anchor[0], point[1] - anchor[1]) >
                        VERTEX_DRAG_THRESHOLD;
                      if (!moved) return current;
                      return {
                        ...current,
                        draggedIndex: current.pendingVertexIndex,
                        pendingVertexIndex: null,
                      };
                    }
                    if (current.draggedIndex === null) return current;
                    const nextPoints = [...current.points];
                    nextPoints[current.draggedIndex] = point;
                    return { ...current, points: nextPoints };
                  });
                }}
                onInsertVertexOnEdge={(nextPoints) => {
                  if (!selectedSegmentId) return;
                  const pending = {
                    segmentId: selectedSegmentId,
                    points: nextPoints,
                    draggedIndex: null as number | null,
                    pendingVertexIndex: null as number | null,
                  };
                  setVertexEdit(pending);
                  void Promise.resolve(
                    onSegmentPointsChange(pending.segmentId, pending.points),
                  ).finally(() => {
                    setVertexEdit((latest) =>
                      latest?.segmentId === pending.segmentId ? null : latest,
                    );
                  });
                }}
              />
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
