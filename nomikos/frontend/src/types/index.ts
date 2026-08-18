import type { LayoutPoint } from "../api/client";

/** Shared types for the public document canvas overlay. */
export type PointTuple = LayoutPoint;

export interface Region {
  id: number;
  boundary: PointTuple[];
  bbox: [number, number, number, number];
}
