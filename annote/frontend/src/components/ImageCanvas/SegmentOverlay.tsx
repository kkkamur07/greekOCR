"use client";

import type { Segment } from "@/types/api";

interface SegmentOverlayProps {
  imageWidth: number;
  imageHeight: number;
  segments: Segment[];
  selectedId: string | null;
  draftPoints: [number, number][];
  editMode: boolean;
  visible: boolean;
  interactive: boolean;
  /** Canvas zoom scale — vertex handles shrink in image space so they stay small on screen. */
  zoomScale: number;
  onSelect: (id: string) => void;
  clientToImage: (clientX: number, clientY: number) => [number, number] | null;
  onVertexDrag: (segmentId: string, pointIndex: number, x: number, y: number) => void;
}

const HANDLE_RADIUS = 6;
const DRAFT_RADIUS = 4;
const LABEL_FONT_SIZE = 14;

function screenScaled(size: number, zoomScale: number): number {
  return size / Math.max(zoomScale, 0.05);
}

function segmentPath(points: [number, number][]): string {
  if (points.length === 0) return "";
  return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ") + " Z";
}

export default function SegmentOverlay({
  imageWidth,
  imageHeight,
  segments,
  selectedId,
  draftPoints,
  editMode,
  visible,
  interactive,
  zoomScale,
  onSelect,
  clientToImage,
  onVertexDrag,
}: SegmentOverlayProps) {
  if (!visible) return null;

  const handleRadius = screenScaled(HANDLE_RADIUS, zoomScale);
  const draftRadius = screenScaled(DRAFT_RADIUS, zoomScale);
  const labelFontSize = screenScaled(LABEL_FONT_SIZE, zoomScale);
  const handleStroke = screenScaled(2, zoomScale);
  const segmentStroke = screenScaled(1.5, zoomScale);
  const selectedStroke = screenScaled(2.5, zoomScale);

  return (
    <svg
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      className="absolute inset-0 h-full w-full"
      style={{ pointerEvents: interactive ? "auto" : "none" }}
    >
      {segments.map((seg) => {
        const selected = seg.id === selectedId;
        return (
          <g key={seg.id}>
            <path
              data-segment-hit=""
              data-segment-id={seg.id}
              d={segmentPath(seg.points)}
              fill={selected ? "rgba(59,130,246,0.25)" : "rgba(34,197,94,0.15)"}
              stroke={selected ? "#2563eb" : "#16a34a"}
              strokeWidth={selected ? selectedStroke : segmentStroke}
              style={{ pointerEvents: interactive ? "auto" : "none", cursor: interactive ? "pointer" : "default" }}
              onPointerDown={(e) => {
                if (!interactive) return;
                e.stopPropagation();
              }}
              onPointerUp={(e) => {
                if (!interactive) return;
                e.stopPropagation();
                onSelect(seg.id);
              }}
              onClick={(e) => e.stopPropagation()}
            />
            {seg.points.length > 0 && (
              <text
                x={seg.points[0][0] + 4}
                y={seg.points[0][1] - 6}
                fill={selected ? "#1d4ed8" : "#15803d"}
                fontSize={labelFontSize}
                fontWeight={600}
                style={{ pointerEvents: "none" }}
              >
                {seg.number}
              </text>
            )}
            {editMode &&
              selected &&
              seg.points.map((pt, idx) => (
                <circle
                  key={idx}
                  data-vertex-handle=""
                  cx={pt[0]}
                  cy={pt[1]}
                  r={handleRadius}
                  fill="#fff"
                  stroke="#2563eb"
                  strokeWidth={handleStroke}
                  className="cursor-move"
                  style={{ pointerEvents: "auto" }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    const handle = e.currentTarget;
                    handle.setPointerCapture(e.pointerId);
                    const move = (ev: PointerEvent) => {
                      const coords = clientToImage(ev.clientX, ev.clientY);
                      if (coords) onVertexDrag(seg.id, idx, coords[0], coords[1]);
                    };
                    const up = () => {
                      try {
                        handle.releasePointerCapture(e.pointerId);
                      } catch {
                        /* capture may already be released */
                      }
                      window.removeEventListener("pointermove", move);
                      window.removeEventListener("pointerup", up);
                    };
                    window.addEventListener("pointermove", move);
                    window.addEventListener("pointerup", up);
                  }}
                />
              ))}
          </g>
        );
      })}
      {draftPoints.length > 0 && (
        <g style={{ pointerEvents: "none" }}>
          <polyline
            points={draftPoints.map((p) => p.join(",")).join(" ")}
            fill="none"
            stroke="#f59e0b"
            strokeWidth={screenScaled(2, zoomScale)}
            strokeDasharray={`${screenScaled(6, zoomScale)} ${screenScaled(4, zoomScale)}`}
          />
          {draftPoints.map((pt, idx) => (
            <circle key={idx} cx={pt[0]} cy={pt[1]} r={draftRadius} fill="#f59e0b" />
          ))}
        </g>
      )}
    </svg>
  );
}
