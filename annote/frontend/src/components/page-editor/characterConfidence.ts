import type { LineTranscriptionResponse } from '../../api/client';

export type CharacterConfidence = {
  char: string;
  confidence: number;
};

/** Optional per-character scores until OpenAPI exposes them on LineTranscriptionResponse. */
export type LineTranscriptionWithCharacterConfidence = LineTranscriptionResponse & {
  character_confidences?: CharacterConfidence[] | null;
};

export function confidenceHighlightColor(confidence: number): string {
  if (confidence > 0.9) return '#b7eb8f';
  if (confidence > 0.7) return '#91caff';
  if (confidence > 0.5) return '#ffe58f';
  return '#ffa39e';
}

export function confidenceLabelColor(confidence: number): string {
  if (confidence > 0.9) return '#389e0d';
  if (confidence > 0.7) return '#0958d9';
  if (confidence > 0.5) return '#d48806';
  return '#cf1322';
}

export function formatConfidencePercent(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function characterConfidencesForTranscription(
  transcription: LineTranscriptionWithCharacterConfidence,
): CharacterConfidence[] {
  const explicit = transcription.character_confidences;
  if (explicit && explicit.length === transcription.text.length) {
    return explicit;
  }
  if (transcription.confidence === null) {
    return transcription.text.split('').map((char) => ({ char, confidence: 0 }));
  }
  return transcription.text.split('').map((char) => ({
    char,
    confidence: transcription.confidence as number,
  }));
}

export function hasDistinctCharacterConfidences(
  transcription: LineTranscriptionWithCharacterConfidence,
): boolean {
  const explicit = transcription.character_confidences;
  return Boolean(explicit && explicit.length === transcription.text.length);
}
