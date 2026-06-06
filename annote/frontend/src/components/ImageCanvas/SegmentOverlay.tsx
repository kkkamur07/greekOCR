"use client";

import type { Segment } from "@/types/api";

interface SegmentOverlayProps {
  segments: Segment[];
  selectedId: string | null;
  draftPoints: [number, number][];
  editMode: boolean;
  visible: boolean;
  interactive: boolean;
  onSelect: (id: string) => void;
  onVertexDrag: (segmentId: string, pointIndex: number, x: number, y: number) => void;
}

function segmentPath(points: [number, number][]): string {
  if (points.length === 0) return "";
  return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ") + " Z";
}

export default function SegmentOverlay({
  segments,
  selectedId,
  draftPoints,
  editMode,
  visible,
  interactive,
  onSelect,
  onVertexDrag,
}: SegmentOverlayProps) {
  if (!visible) return null;

  return (
    <svg className="absolute inset-0 h-full w-full" style={{ pointerEvents: interactive ? "auto" : "none" }}>
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
              strokeWidth={selected ? 2.5 : 1.5}
              vectorEffect="non-scaling-stroke"
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
                fontSize={14}
                fontWeight={600}
                style={{ pointerEvents: "none" }}
              >
                {seg.number}
              </text>
            )}
            {editMode &&
              selected &&
              interactive &&
              seg.points.map((pt, idx) => (
                <circle
                  key={idx}
                  cx={pt[0]}
                  cy={pt[1]}
                  r={6}
                  fill="#fff"
                  stroke="#2563eb"
                  strokeWidth={2}
                  className="cursor-move"
                  style={{ pointerEvents: "auto" }}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                  }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    const svg = (e.target as SVGElement).ownerSVGElement;
                    if (!svg) return;
                    const move = (ev: PointerEvent) => {
                      const rect = svg.getBoundingClientRect();
                      const scaleX = svg.clientWidth / rect.width;
                      const scaleY = svg.clientHeight / rect.height;
                      onVertexDrag(seg.id, idx, (ev.clientX - rect.left) * scaleX, (ev.clientY - rect.top) * scaleY);
                    };
                    const up = () => {
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
            strokeWidth={2}
            strokeDasharray="6 4"
            vectorEffect="non-scaling-stroke"
          />
          {draftPoints.map((pt, idx) => (
            <circle key={idx} cx={pt[0]} cy={pt[1]} r={4} fill="#f59e0b" />
          ))}
        </g>
      )}
    </svg>
  );
}
