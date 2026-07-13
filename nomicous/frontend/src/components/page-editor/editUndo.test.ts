import { describe, expect, it } from "vitest";

import type { LineResponse } from "../../api/client";
import {
  applyCanvasEdit,
  applyCanvasEditInverse,
  pushEditOntoStack,
  type CanvasEdit,
} from "./editUndo";

const baseLine = (overrides: Partial<LineResponse> = {}): LineResponse => ({
  id: "line-1",
  part_id: "part-1",
  block_id: null,
  order: 0,
  kind: "polygon",
  points: [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
  ],
  source: "manual",
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [],
  created_at: "2026-06-16T10:00:00Z",
  ...overrides,
});

describe("editUndo", () => {
  it("applies and inverts point edits", () => {
    const before = baseLine().points;
    const after: [number, number][] = [
      [0, 0],
      [20, 0],
      [10, 10],
      [0, 10],
    ];
    const edit: CanvasEdit = {
      kind: "points",
      segmentId: "line-1",
      before,
      after,
    };
    const lines = [baseLine()];
    const forward = applyCanvasEdit(lines, edit);
    expect(forward[0].points).toEqual(after);
    expect(applyCanvasEditInverse(forward, edit)[0].points).toEqual(before);
  });

  it("applies and inverts create / delete edits", () => {
    const created = baseLine({ id: "line-2", order: 1 });
    const createEdit: CanvasEdit = { kind: "create", line: created };
    const withCreated = applyCanvasEdit([baseLine()], createEdit);
    expect(withCreated.map((line) => line.id)).toEqual(["line-1", "line-2"]);
    expect(applyCanvasEditInverse(withCreated, createEdit)).toEqual([
      baseLine(),
    ]);

    const deleteEdit: CanvasEdit = { kind: "delete", line: created };
    const without = applyCanvasEdit(withCreated, deleteEdit);
    expect(without.map((line) => line.id)).toEqual(["line-1"]);
    expect(
      applyCanvasEditInverse(without, deleteEdit).map((line) => line.id),
    ).toEqual(["line-1", "line-2"]);
  });

  it("caps the undo stack size", () => {
    const edits = Array.from({ length: 3 }, (_, index) => ({
      kind: "points" as const,
      segmentId: "line-1",
      before: baseLine().points,
      after: baseLine().points.map(
        ([x, y]) => [x + index, y] as [number, number],
      ),
    }));
    let stack: CanvasEdit[] = [];
    for (const edit of edits) {
      stack = pushEditOntoStack(stack, edit, 2);
    }
    expect(stack).toHaveLength(2);
    expect(stack[0]).toEqual(edits[1]);
    expect(stack[1]).toEqual(edits[2]);
  });
});
