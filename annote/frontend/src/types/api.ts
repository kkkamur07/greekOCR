import type { components } from "./openapi";

type Schemas = components["schemas"];

export type PairingProgress = Schemas["PairingProgress"];
export type PageSummary = Schemas["PageSummary"];
export type PageListResponse = Schemas["PageListResponse"];
export type TextLine = Schemas["TextLineOut"];
export type TranscriptionResponse = Schemas["TranscriptionResponse"];
export type SegmentKind = Schemas["Segment"]["kind"];
export type Segment = Omit<Schemas["Segment"], "points"> & {
  points: [number, number][];
};
export type ExportMetadata = Schemas["ExportMetadata"];
export type PageAnnotation = {
  segments: Segment[];
  export_metadata: ExportMetadata | null;
  locked: boolean;
};
export type HistorySnapshotSummary = Schemas["HistorySnapshotSummary"];
export type HistoryListResponse = Schemas["HistoryListResponse"];
export type ExportResponse = Schemas["ExportResponse"];
export type AutoSegmentRequest = Schemas["AutoSegmentRequest"];

export type ExportStep = "rectify" | "save";

export interface ExportProgressEvent {
  type: "progress";
  current: number;
  total: number;
  segment_number: number;
  step: ExportStep;
}

export type DrawTool = "pan" | "select" | "polygon" | "rectangle";
