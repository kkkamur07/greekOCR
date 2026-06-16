export interface Region {
  id: number;
  boundary: number[][];
  bbox: [number, number, number, number];
}

export interface Transcription {
  region_id: number;
  text: string;
  confidence: number;
}

export interface UploadResponse {
  image_id: string;
  width: number;
  height: number;
  message: string;
}

export interface SegmentResponse {
  image_id: string;
  regions: Region[];
  total_regions: number;
}

export interface TranscribeResponse {
  image_id: string;
  transcriptions: Transcription[];
}

export interface OCRState {
  imageId: string | null;
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  regions: Region[];
  transcriptions: Transcription[];
  selectedRegionId: number | null;
  isLoading: boolean;
  error: string | null;
}

// New types for enhanced interaction
export type DrawMode = 'none' | 'box' | 'polygon';
export type EditMode = 'none' | 'vertices';

export interface Point {
  x: number;
  y: number;
}

// History types for undo/redo
export interface HistoryState {
  regions: Region[];
  transcriptions: Transcription[];
  selectedRegionId: number | null;
}

export interface Action {
  type: 'ADD_REGION' | 'UPDATE_REGION' | 'DELETE_REGION' | 'TRANSCRIBE' | 'UPDATE_TRANSCRIPTION';
  timestamp: number;
  description: string;
}

// Editing settings
export interface EditingSettings {
  showBoundingBoxes: boolean;
  vertexSize: number; // 1-3 scale
  moveStep: number; // pixels to move with arrow keys
}

export type DrawMode = 'none' | 'box' | 'polygon';
export type EditMode = 'none' | 'move' | 'vertices';

export interface Point {
  x: number;
  y: number;
}