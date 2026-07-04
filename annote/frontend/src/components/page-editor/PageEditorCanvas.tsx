import { useState, type MouseEvent } from 'react';
import { TransformComponent, TransformWrapper } from 'react-zoom-pan-pinch';
import type { LinePoint, LineResponse, PartLayoutResponse } from '../../api/client';
import { AuthenticatedImage } from '../AuthenticatedImage';
import { points } from './canvasGeometry';

type CanvasSurfaceProps = {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  drawingRectangle: boolean;
  drawingPolygon: boolean;
  onDraftStart: (point: LinePoint) => void;
  onRectangleDrawn: (point: LinePoint) => void;
  onPolygonPoint: (point: LinePoint) => void;
  onPolygonComplete: () => void;
  onSelectLine: (lineId: string) => void;
  onSelectSegment: (lineId: string) => void;
};

function CanvasSurface({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  drawingRectangle,
  drawingPolygon,
  onDraftStart,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
}: CanvasSurfaceProps) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const canvasWidth = naturalSize?.width ?? imageWidth;
  const canvasHeight = naturalSize?.height ?? imageHeight;

  const eventPoint = (event: MouseEvent<SVGSVGElement>): LinePoint => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return [event.clientX, event.clientY];
    }
    return [
      Math.round(((event.clientX - rect.left) / rect.width) * canvasWidth),
      Math.round(((event.clientY - rect.top) / rect.height) * canvasHeight),
    ];
  };

  return (
    <div
      style={{
        position: 'relative',
        width: canvasWidth,
        height: canvasHeight,
        background: '#fff',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
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
          if (drawingRectangle) onDraftStart(eventPoint(event));
        }}
        onMouseUp={(event) => {
          if (drawingRectangle) onRectangleDrawn(eventPoint(event));
        }}
        onClick={(event) => {
          if (drawingPolygon) onPolygonPoint(eventPoint(event));
        }}
        onDoubleClick={() => {
          if (drawingPolygon) onPolygonComplete();
        }}
        style={{
          position: 'absolute',
          inset: 0,
          width: canvasWidth,
          height: canvasHeight,
          cursor: drawingRectangle || drawingPolygon ? 'crosshair' : 'grab',
        }}
      >
        {layout.blocks.map((block) => (
          <polygon
            key={block.id}
            aria-label={`Block ${block.id}`}
            points={points(block.box)}
            fill="rgba(216, 199, 161, 0.08)"
            stroke="#d8c7a1"
            strokeWidth={4}
          />
        ))}
        {layout.lines.map((line) => (
          <polyline
            key={line.id}
            aria-label={`Line ${line.id} baseline`}
            onClick={() => onSelectLine(line.id)}
            points={points(line.baseline)}
            fill="none"
            stroke={line.manual_geometry ? '#58d68d' : '#5dade2'}
            strokeLinecap="round"
            strokeWidth={6}
          />
        ))}
        {lines
          .slice()
          .sort((a, b) => a.order - b.order)
          .map((line) => (
            <polygon
              key={line.id}
              aria-label={`Segment ${line.order + 1}`}
              onClick={() => onSelectSegment(line.id)}
              points={points(line.points)}
              fill="rgba(88, 214, 141, 0.14)"
              stroke="#58d68d"
              strokeWidth={5}
            />
          ))}
      </svg>
    </div>
  );
}

export function PageEditorCanvas({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  drawingRectangle,
  drawingPolygon,
  onDraftStart,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
}: CanvasSurfaceProps) {
  const [zoomLevel, setZoomLevel] = useState(0.75);
  const canvasWidth = imageWidth;
  const canvasHeight = imageHeight;

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#f5f5f5' }}>
      <TransformWrapper
        initialScale={0.75}
        minScale={0.2}
        maxScale={6}
        centerOnInit
        wheel={{ step: 0.12, smoothStep: 0.002 }}
        doubleClick={{ disabled: drawingRectangle || drawingPolygon, step: 0.7 }}
        panning={{ disabled: drawingRectangle || drawingPolygon, velocityDisabled: false }}
        pinch={{ disabled: drawingRectangle || drawingPolygon }}
        onTransformed={(ref) => setZoomLevel(ref.state.scale)}
      >
        {({ zoomIn, zoomOut, resetTransform }) => (
          <>
            <div
              style={{
                position: 'absolute',
                top: 12,
                right: 12,
                zIndex: 10,
                display: 'flex',
                gap: 4,
                padding: 4,
                borderRadius: 8,
                background: '#fff',
                boxShadow: '0 2px 12px rgba(0, 0, 0, 0.15)',
              }}
            >
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => zoomOut()}>
                -
              </button>
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => resetTransform()}>
                {Math.round(zoomLevel * 100)}%
              </button>
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => zoomIn()}>
                +
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
                drawingRectangle={drawingRectangle}
                drawingPolygon={drawingPolygon}
                onDraftStart={onDraftStart}
                onRectangleDrawn={onRectangleDrawn}
                onPolygonPoint={onPolygonPoint}
                onPolygonComplete={onPolygonComplete}
                onSelectLine={onSelectLine}
                onSelectSegment={onSelectSegment}
              />
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
