import { describe, expect, it } from "vitest";

import {
  canvasStrokeWidth,
  cleanPolygonPoints,
  findPolygonEdgeInsert,
  insertPolygonVertexAtClick,
  normalizeGeometryPoints,
  offsetGeometry,
  points,
  removePolygonVertex,
  withGeometryPoints,
} from "./canvasGeometry";

describe("canvasGeometry", () => {
  it("formats plain point arrays for SVG", () => {
    expect(
      points([
        [1, 2],
        [3, 4],
      ]),
    ).toBe("1,2 3,4");
  });

  it("normalizes backend box and baseline objects with points", () => {
    expect(
      points({
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
      }),
    ).toBe("0,0 10,0 10,10");

    expect(
      normalizeGeometryPoints({
        points: [
          [2, 2],
          [22, 2],
        ],
      }),
    ).toEqual([
      [2, 2],
      [22, 2],
    ]);
  });

  it("normalizes GeoJSON line coordinates", () => {
    expect(
      points({
        type: "LineString",
        coordinates: [
          [1, 1],
          [2, 1],
        ],
      }),
    ).toBe("1,1 2,1");
  });

  it("offsets geometry while preserving object shape", () => {
    expect(
      offsetGeometry(
        {
          points: [
            [2, 2],
            [22, 2],
          ],
        },
        5,
      ),
    ).toEqual({
      points: [
        [2, 7],
        [22, 7],
      ],
    });
  });

  it("writes updated points back into geometry objects", () => {
    expect(
      withGeometryPoints({ type: "LineString", coordinates: [[1, 1]] }, [
        [1, 6],
        [2, 6],
      ]),
    ).toEqual({
      type: "LineString",
      coordinates: [
        [1, 6],
        [2, 6],
      ],
    });
  });

  it("scales stroke width inversely with zoom for screen consistency", () => {
    const zoomedOut = canvasStrokeWidth(2, 0.5, 1.5, 2000);
    const zoomedIn = canvasStrokeWidth(2, 2, 1.5, 2000);
    expect(zoomedIn).toBeLessThan(zoomedOut);
    expect(canvasStrokeWidth(2, 1, 2, 2000)).toBeGreaterThan(
      canvasStrokeWidth(2, 1, 1, 2000),
    );
  });

  it("finds the nearest polygon edge for vertex insertion", () => {
    const square: [number, number][] = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    expect(findPolygonEdgeInsert(square, [50, 5], 12)).toEqual({
      insertIndex: 1,
      point: [50, 0],
    });
    expect(findPolygonEdgeInsert(square, [50, 50], 12)).toBeNull();
    expect(findPolygonEdgeInsert(square, [0, 0], 12)).toBeNull();
  });

  it("inserts a vertex on the clicked edge", () => {
    const triangle: [number, number][] = [
      [0, 0],
      [100, 0],
      [50, 80],
    ];
    expect(insertPolygonVertexAtClick(triangle, [50, 2], 12)).toEqual([
      [0, 0],
      [50, 0],
      [100, 0],
      [50, 80],
    ]);
  });

  it("removes consecutive vertices closer than the minimum spacing", () => {
    expect(
      cleanPolygonPoints([
        [0, 0],
        [1, 0],
        [50, 0],
        [100, 0],
        [100, 50],
        [0, 50],
      ]),
    ).toEqual([
      [0, 0],
      [50, 0],
      [100, 0],
      [100, 50],
      [0, 50],
    ]);
  });

  it("removes a polygon vertex while keeping at least three points", () => {
    const square: [number, number][] = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    const triangle: [number, number][] = [
      [0, 0],
      [100, 0],
      [50, 80],
    ];
    expect(removePolygonVertex(square, 1)).toEqual([
      [0, 0],
      [100, 100],
      [0, 100],
    ]);
    expect(removePolygonVertex(triangle, 0)).toBeNull();
  });
});
