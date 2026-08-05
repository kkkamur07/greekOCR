import { describe, expect, it } from "vitest";

import {
  nextSegmentOrder,
  segmentNumberFor,
  segmentNumbersById,
  segmentsInNumberOrder,
} from "./segmentNumbering";

function page(...orders: number[]) {
  return orders.map((order) => ({ id: `line-${order}`, order }));
}

describe("segmentNumbering", () => {
  it("numbers the Segments of a Page from 1 in stored order", () => {
    expect(segmentNumbersById(page(0, 1, 2))).toEqual(
      new Map([
        ["line-0", 1],
        ["line-1", 2],
        ["line-2", 3],
      ]),
    );
  });

  it("numbers densely when a delete has left a gap in the orders", () => {
    const lines = page(0, 1, 3);

    expect(segmentNumberFor(lines, "line-3")).toBe(3);
    expect(segmentNumbersById(lines)).toEqual(
      new Map([
        ["line-0", 1],
        ["line-1", 2],
        ["line-3", 3],
      ]),
    );
  });

  it("numbers by stored order, not by the order the API listed", () => {
    const lines = [
      { id: "line-c", order: 3 },
      { id: "line-a", order: 0 },
      { id: "line-b", order: 1 },
    ];

    expect(segmentsInNumberOrder(lines).map((line) => line.id)).toEqual([
      "line-a",
      "line-b",
      "line-c",
    ]);
    expect(segmentNumberFor(lines, "line-b")).toBe(2);
  });

  it("orders colliding orders the same way whatever the input order", () => {
    const collided = [
      { id: "line-b", order: 3 },
      { id: "line-a", order: 3 },
    ];

    expect(segmentsInNumberOrder(collided).map((line) => line.id)).toEqual([
      "line-a",
      "line-b",
    ]);
    expect(
      segmentsInNumberOrder([...collided].reverse()).map((line) => line.id),
    ).toEqual(["line-a", "line-b"]);
  });

  it("leaves the input array untouched", () => {
    const lines = page(2, 0, 1);
    segmentsInNumberOrder(lines);

    expect(lines.map((line) => line.id)).toEqual([
      "line-2",
      "line-0",
      "line-1",
    ]);
  });

  it("has no number for a Segment that is not on the Page", () => {
    expect(segmentNumberFor(page(0, 1), "line-9")).toBeNull();
    expect(segmentNumberFor(page(0, 1), null)).toBeNull();
    expect(segmentNumberFor([], "line-0")).toBeNull();
  });

  it("numbers the only Segment of a single-Segment Page 1", () => {
    expect(segmentNumberFor(page(0), "line-0")).toBe(1);
    expect(segmentNumberFor(page(7), "line-7")).toBe(1);
  });

  it("gives the first Segment drawn on an empty Page order 0", () => {
    expect(nextSegmentOrder([])).toBe(0);
  });

  it("gives a new Segment an order past every Segment already stored", () => {
    expect(nextSegmentOrder(page(0))).toBe(1);
    expect(nextSegmentOrder(page(0, 1, 2))).toBe(3);
  });

  it("does not reuse an order freed by a delete", () => {
    const lines = page(0, 1, 3);

    expect(nextSegmentOrder(lines)).toBe(4);
    expect(lines.some((line) => line.order === nextSegmentOrder(lines))).toBe(
      false,
    );
  });
});
