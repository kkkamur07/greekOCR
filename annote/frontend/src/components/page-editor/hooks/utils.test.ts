import { describe, expect, it } from 'vitest';

import type { LineResponse, TranscriptionLayerResponse } from '../../../api/client';
import {
  modelLayerIdForPromotion,
  showsModelSourceReview,
  transcriptionForOcrReview,
} from './utils';

const MODEL_LAYER: TranscriptionLayerResponse = {
  id: 'model-1',
  document_id: 'doc-1',
  name: 'Kraken run',
  kind: 'model',
  created_by_job_id: 'job-1',
  created_at: '2026-06-16T10:01:00Z',
};

const GROUND_TRUTH_LAYER: TranscriptionLayerResponse = {
  id: 'ground-truth-1',
  document_id: 'doc-1',
  name: 'Ground truth',
  kind: 'ground_truth',
  created_by_job_id: null,
  created_at: '2026-06-16T10:00:00Z',
};

const LINE = {
  id: 'line-1',
  part_id: 'part-1',
  block_id: null,
  order: 0,
  kind: 'polygon',
  points: [[10, 10], [50, 10], [50, 30], [10, 30]],
  source: 'manual',
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [
    {
      id: 'line-tx-ground-1',
      transcription_id: 'ground-truth-1',
      transcription_kind: 'ground_truth',
      text: 'model suggestion',
      confidence: null,
      text_source: 'model',
    },
    {
      id: 'line-tx-model-1',
      transcription_id: 'model-1',
      transcription_kind: 'model',
      text: 'model suggestion',
      confidence: 0.91,
      text_source: 'model',
    },
  ],
  created_at: '2026-06-16T10:00:00Z',
} as LineResponse;

describe('page editor transcription utils', () => {
  it('shows OCR review for model layers and ground truth with text_source model', () => {
    expect(showsModelSourceReview(LINE.line_transcriptions[1])).toBe(true);
    expect(showsModelSourceReview(LINE.line_transcriptions[0])).toBe(true);
    expect(showsModelSourceReview({ ...LINE.line_transcriptions[0], text_source: 'human_edited' })).toBe(false);
  });

  it('selects model transcription for OCR review on ground truth when text_source is model', () => {
    expect(transcriptionForOcrReview(LINE, GROUND_TRUTH_LAYER)).toEqual(LINE.line_transcriptions[1]);
    expect(transcriptionForOcrReview(LINE, MODEL_LAYER)).toEqual(LINE.line_transcriptions[1]);
  });

  it('resolves the model layer id used for ground truth promotion', () => {
    expect(modelLayerIdForPromotion(LINE, MODEL_LAYER)).toBe('model-1');
    expect(modelLayerIdForPromotion(LINE, GROUND_TRUTH_LAYER)).toBe('model-1');
  });
});
