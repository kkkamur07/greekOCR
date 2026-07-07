import type { LinePoint, LineResponse, PartLayoutResponse } from '../../api/client';

export type PageEditorCanvasProps = {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  drawingRectangle: boolean;
  drawingPolygon: boolean;
  onDraftStart: (point: LinePoint) => void;
  onRectangleDrawn: (point: LinePoint) => void;
  onPolygonPoint: (point: LinePoint) => void;
  onPolygonComplete: () => void;
  onSelectLine: (lineId: string) => void;
  onSelectSegment: (lineId: string) => void;
};
