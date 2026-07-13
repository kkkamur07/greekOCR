import type { LinePoint, LineResponse } from "../../api/client";

/** One in-session canvas step for Edit undo / Edit redo (not Annotation history). */
export type CanvasEdit =
  | {
      kind: "points";
      segmentId: string;
      before: LinePoint[];
      after: LinePoint[];
    }
  | {
      kind: "create";
      line: LineResponse;
    }
  | {
      kind: "delete";
      line: LineResponse;
    };

export function applyCanvasEdit(
  lines: LineResponse[],
  edit: CanvasEdit,
): LineResponse[] {
  switch (edit.kind) {
    case "points":
      return lines.map((line) =>
        line.id === edit.segmentId
          ? { ...line, points: edit.after, source: "manual" as const }
          : line,
      );
    case "create":
      return mergeLine(lines, edit.line);
    case "delete":
      return lines.filter((line) => line.id !== edit.line.id);
  }
}

export function applyCanvasEditInverse(
  lines: LineResponse[],
  edit: CanvasEdit,
): LineResponse[] {
  switch (edit.kind) {
    case "points":
      return lines.map((line) =>
        line.id === edit.segmentId
          ? { ...line, points: edit.before, source: "manual" as const }
          : line,
      );
    case "create":
      return lines.filter((line) => line.id !== edit.line.id);
    case "delete":
      return mergeLine(lines, edit.line);
  }
}

function mergeLine(lines: LineResponse[], saved: LineResponse): LineResponse[] {
  const without = lines.filter((line) => line.id !== saved.id);
  return [...without, saved].sort((a, b) => a.order - b.order);
}

export function pushEditOntoStack(
  stack: CanvasEdit[],
  edit: CanvasEdit,
  maxSize = 50,
): CanvasEdit[] {
  const next = [...stack, edit];
  if (next.length <= maxSize) return next;
  return next.slice(next.length - maxSize);
}
