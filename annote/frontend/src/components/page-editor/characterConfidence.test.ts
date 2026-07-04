import { describe, expect, it } from 'vitest';

import {
  characterConfidencesForTranscription,
  confidenceHighlightColor,
  confidenceLabelColor,
  formatConfidencePercent,
  hasDistinctCharacterConfidences,
  type LineTranscriptionWithCharacterConfidence,
} from './characterConfidence';

const BASE_TRANSCRIPTION: LineTranscriptionWithCharacterConfidence = {
  id: 'tx-1',
  transcription_id: 'model-1',
  transcription_kind: 'model',
  text: 'abc',
  confidence: 0.82,
  text_source: 'model',
};

describe('characterConfidence', () => {
  it('maps confidence values to highlight and label colors', () => {
    expect(confidenceHighlightColor(0.95)).toBe('#b7eb8f');
    expect(confidenceHighlightColor(0.8)).toBe('#91caff');
    expect(confidenceHighlightColor(0.6)).toBe('#ffe58f');
    expect(confidenceHighlightColor(0.2)).toBe('#ffa39e');
    expect(confidenceLabelColor(0.95)).toBe('#389e0d');
    expect(formatConfidencePercent(0.825)).toBe('82.5%');
  });

  it('uses explicit per-character confidences when aligned with text', () => {
    const transcription: LineTranscriptionWithCharacterConfidence = {
      ...BASE_TRANSCRIPTION,
      character_confidences: [
        { char: 'a', confidence: 0.99 },
        { char: 'b', confidence: 0.55 },
        { char: 'c', confidence: 0.71 },
      ],
    };

    expect(hasDistinctCharacterConfidences(transcription)).toBe(true);
    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: 'a', confidence: 0.99 },
      { char: 'b', confidence: 0.55 },
      { char: 'c', confidence: 0.71 },
    ]);
  });

  it('falls back to line confidence for each character when per-char scores are missing', () => {
    expect(hasDistinctCharacterConfidences(BASE_TRANSCRIPTION)).toBe(false);
    expect(characterConfidencesForTranscription(BASE_TRANSCRIPTION)).toEqual([
      { char: 'a', confidence: 0.82 },
      { char: 'b', confidence: 0.82 },
      { char: 'c', confidence: 0.82 },
    ]);
  });
});
