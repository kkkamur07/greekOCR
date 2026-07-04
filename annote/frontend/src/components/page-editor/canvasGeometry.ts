import type { LayoutPoint, LinePoint } from '../../api/client';

export function points(points: LayoutPoint[] | LinePoint[] | null | undefined): string {
  return (points ?? []).map(([x, y]) => `${x},${y}`).join(' ');
}

export function rectanglePoints(start: LinePoint, end: LinePoint): LinePoint[] {
  const [startX, startY] = start;
  const [endX, endY] = end;
  return [
    [Math.min(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.max(startY, endY)],
    [Math.min(startX, endX), Math.max(startY, endY)],
  ];
}
