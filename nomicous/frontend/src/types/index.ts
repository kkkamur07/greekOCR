/** Shared types for the public document canvas overlay. */
export type PointTuple = [number, number];

export interface Region {
  id: number;
  boundary: PointTuple[];
  bbox: [number, number, number, number];
}
