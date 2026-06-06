export interface PageSummary {
  stem: string;
  has_transcription: boolean;
  segment_count: number;
  export_dirty: boolean;
}

export interface PageListResponse {
  pages: PageSummary[];
}

export interface TextLine {
  index: number;
  text: string;
}

export interface TranscriptionResponse {
  raw_text: string | null;
  text_lines: TextLine[];
  status: string;
}

export type SegmentKind = "polygon" | "rectangle";

export interface Segment {
  id: string;
  number: number;
  kind: SegmentKind;
  points: [number, number][];
  paired_text_line_index: number | null;
  text_override?: string | null;
}

export interface ExportMetadata {
  exported_at: string;
  content_hash: string;
}

export interface PageAnnotation {
  segments: Segment[];
  export_metadata: ExportMetadata | null;
}

export interface ExportWarnings {
  unpaired_segments: number[];
  unused_text_lines: number[];
}

export interface ExportResponse {
  exported_count: number;
  warnings: ExportWarnings;
  steps: string[];
}

export type ExportStep = "rectify" | "binarize" | "save";

export interface ExportProgressEvent {
  type: "progress";
  current: number;
  total: number;
  segment_number: number;
  step: ExportStep;
}

export type DrawTool = "pan" | "select" | "polygon" | "rectangle";
