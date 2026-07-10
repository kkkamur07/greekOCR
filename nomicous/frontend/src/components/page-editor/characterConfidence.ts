import type { LineTranscriptionResponse } from "../../api/client";

export type CharacterConfidence = {
  char: string;
  confidence: number;
};

/** Optional per-character scores until OpenAPI exposes them on LineTranscriptionResponse. */
export type LineTranscriptionWithCharacterConfidence =
  LineTranscriptionResponse & {
    character_confidences?: CharacterConfidence[] | null;
    text_source?: string | null;
  };

export function confidenceTierClass(confidence: number): string {
  if (confidence > 0.9) return "ch-high";
  if (confidence > 0.5) return "ch-mid";
  return "ch-low";
}

/** Human-readable tier for aria / tooltips (matches strip legend). */
export function confidenceTierLabel(confidence: number): string {
  if (confidence > 0.9) return "high";
  if (confidence > 0.5) return "mid";
  return "low";
}

export function confidenceHighlightColor(confidence: number): string {
  if (confidence > 0.9) return "#059669";
  if (confidence > 0.7) return "#d97706";
  if (confidence > 0.5) return "#d97706";
  return "#dc2626";
}

export function confidenceLabelColor(confidence: number): string {
  if (confidence > 0.9) return "#059669";
  if (confidence > 0.7) return "#d97706";
  if (confidence > 0.5) return "#d97706";
  return "#dc2626";
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
    return transcription.text
      .split("")
      .map((char) => ({ char, confidence: 0 }));
  }
  return transcription.text.split("").map((char) => ({
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
