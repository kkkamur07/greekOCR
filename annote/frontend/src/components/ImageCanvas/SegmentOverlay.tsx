"use client";

import type { Segment } from "@/types/api";
import { segmentIsPaired } from "@/lib/pairingProgress";

interface SegmentOverlayProps {
  imageWidth: number;
  imageHeight: number;
  segments: Segment[];
  selectedId: string | null;
  draftPoints: [number, number][];
  editMode: boolean;
  visible: boolean;
  interactive: boolean;
  /** When true, show dashed Kraken ceiling for the selected Kraken segment. */
  showKrakenCeiling: boolean;
  /** Canvas zoom scale — vertex handles shrink in image space so they stay small on screen. */
  zoomScale: number;
  onSelect: (id: string) => void;
  clientToImage: (clientX: number, clientY: number) => [number, number] | null;
  selectedVertexIndex: number | null;
  onSelectVertex: (index: number | null) => void;
  onVertexDrag: (segmentId: string, pointIndex: number, x: number, y: number) => void;
  onInsertVertex: (segmentId: string, afterIndex: number, x: number, y: number) => void;
  onRemoveVertex: (segmentId: string, pointIndex: number) => void;
}

export const MIN_SEGMENT_POINTS = 4;

interface SegmentHighlight {
  fill: string;
  stroke: string;
  label: string;
}

const PAIRED_HIGHLIGHT: SegmentHighlight = {
  fill: "rgba(34,197,94,0.15)",
  stroke: "#16a34a",
  label: "#15803d",
};

const UNPAIRED_HIGHLIGHT: SegmentHighlight = {
  fill: "rgba(245,158,11,0.15)",
  stroke: "#d97706",
  label: "#b45309",
};

const SELECTED_HIGHLIGHT: SegmentHighlight = {
  fill: "rgba(59,130,246,0.25)",
  stroke: "#2563eb",
  label: "#1d4ed8",
};

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

function segmentHighlight(selected: boolean, paired: boolean): SegmentHighlight {
  if (selected) return SELECTED_HIGHLIGHT;
  if (paired) return PAIRED_HIGHLIGHT;
  return UNPAIRED_HIGHLIGHT;
}

function segmentDashArray(
  segment: Segment,
  selected: boolean,
  zoomScale: number,
): string | undefined {
  if ((segment.source ?? "manual") !== "kraken" || selected) return undefined;
  const dash = screenScaled(8, zoomScale);
  const gap = screenScaled(5, zoomScale);
  return `${dash} ${gap}`;
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
  showKrakenCeiling,
  zoomScale,
  onSelect,
  clientToImage,
  selectedVertexIndex,
  onSelectVertex,
  onVertexDrag,
  onInsertVertex,
  onRemoveVertex,
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
      style={{ pointerEvents: interactive || editMode ? "auto" : "none" }}
    >
      {segments.map((seg) => {
        const selected = seg.id === selectedId;
        const paired = segmentIsPaired(seg);
        const highlight = segmentHighlight(selected, paired);
        const dashArray = segmentDashArray(seg, selected, zoomScale);
        const ceiling =
          showKrakenCeiling &&
          selected &&
          (seg.source ?? "manual") === "kraken" &&
          seg.kraken_ceiling &&
          seg.kraken_ceiling.length >= 3
            ? seg.kraken_ceiling
            : null;

        return (
          <g key={seg.id}>
            <path
              data-segment-hit=""
              data-segment-id={seg.id}
              d={segmentPath(seg.points)}
              fill={highlight.fill}
              stroke={highlight.stroke}
              strokeWidth={selected ? selectedStroke : segmentStroke}
              strokeDasharray={dashArray}
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
            {ceiling && (
              <path
                data-kraken-ceiling=""
                data-segment-id={seg.id}
                d={segmentPath(ceiling as [number, number][])}
                fill="none"
                stroke="#c026d3"
                strokeWidth={screenScaled(2.5, zoomScale)}
                strokeDasharray={`${screenScaled(12, zoomScale)} ${screenScaled(7, zoomScale)}`}
                style={{ pointerEvents: "none" }}
              />
            )}
            {seg.points.length > 0 && (
              <text
                x={seg.points[0][0] + 4}
                y={seg.points[0][1] - 6}
                fill={highlight.label}
                fontSize={labelFontSize}
                fontWeight={600}
                style={{ pointerEvents: "none" }}
              >
                {seg.number}
              </text>
            )}
            {editMode &&
              selected &&
              seg.points.map((pt, idx) => {
                const next = seg.points[(idx + 1) % seg.points.length];
                return (
                  <line
                    key={`edge-${idx}`}
                    x1={pt[0]}
                    y1={pt[1]}
                    x2={next[0]}
                    y2={next[1]}
                    stroke="transparent"
                    strokeWidth={screenScaled(14, zoomScale)}
                    className="cursor-copy"
                    style={{ pointerEvents: "stroke" }}
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      const coords = clientToImage(e.clientX, e.clientY);
                      if (coords) onInsertVertex(seg.id, idx, coords[0], coords[1]);
                    }}
                  />
                );
              })}
            {editMode &&
              selected &&
              seg.points.map((pt, idx) => {
                const vtxSelected = selectedVertexIndex === idx;
                const canRemove = seg.points.length > MIN_SEGMENT_POINTS;
                return (
                  <circle
                    key={idx}
                    data-vertex-handle=""
                    cx={pt[0]}
                    cy={pt[1]}
                    r={handleRadius}
                    fill={vtxSelected ? "#2563eb" : "#fff"}
                    stroke={vtxSelected ? "#1d4ed8" : "#2563eb"}
                    strokeWidth={vtxSelected ? handleStroke * 1.5 : handleStroke}
                    className={canRemove ? "cursor-move" : "cursor-not-allowed"}
                    style={{ pointerEvents: "auto" }}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      if (!canRemove) return;
                      onRemoveVertex(seg.id, idx);
                    }}
                    onPointerDown={(e) => {
                      if (e.button !== 0) return;
                      e.stopPropagation();
                      e.preventDefault();
                      onSelectVertex(idx);
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
                );
              })}
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
