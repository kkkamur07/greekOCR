import { describe, expect, it } from 'vitest';

import type { LineResponse } from '../../../api/client';

import {
  approvedText,
  lineTextForLayer,
  lineTranscriptionForLayer,
  upsertLineRequest,
  withLocalGroundTruth,
} from './utils';

const BASE_LINE: LineResponse = {
  id: 'line-1',
  part_id: 'part-1',
  block_id: null,
  order: 0,
  kind: 'polygon',
  points: [
    [0, 0],
    [10, 0],
    [10, 5],
    [0, 5],
  ],
  source: 'manual',
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [
    {
      id: 'gt-1',
      transcription_id: 'ground-truth-1',
      transcription_kind: 'ground_truth',
      text: 'approved',
      confidence: null,
    },
    {
      id: 'model-1',
      transcription_id: 'model-layer-1',
      transcription_kind: 'model',
      text: 'model text',
      confidence: 0.9,
    },
  ],
  created_at: '2026-06-16T10:00:00Z',
};

describe('page editor hook utils', () => {
  it('reads approved ground-truth text from a line', () => {
    expect(approvedText(BASE_LINE)).toBe('approved');
    expect(approvedText({ ...BASE_LINE, line_transcriptions: [] })).toBeNull();
  });

  it('reads transcription text and object for a selected layer', () => {
    expect(lineTextForLayer(BASE_LINE, 'model-layer-1')).toBe('model text');
    expect(lineTextForLayer(BASE_LINE, null)).toBe('');
    expect(lineTextForLayer(BASE_LINE, 'missing-layer')).toBe('');
    expect(lineTranscriptionForLayer(BASE_LINE, 'model-layer-1')?.confidence).toBe(0.9);
    expect(lineTranscriptionForLayer(BASE_LINE, 'missing-layer')).toBeNull();
  });

  it('builds an upsert request with optional approved text', () => {
    expect(upsertLineRequest(BASE_LINE, 2)).toEqual({
      id: 'line-1',
      order: 2,
      kind: 'polygon',
      points: BASE_LINE.points,
      source: 'manual',
      approved_text: 'approved',
    });
    expect(upsertLineRequest({ ...BASE_LINE, line_transcriptions: [] }, 0)).toEqual({
      id: 'line-1',
      order: 0,
      kind: 'polygon',
      points: BASE_LINE.points,
      source: 'manual',
    });
  });

  it('updates local ground-truth text for one line', () => {
    const updated = withLocalGroundTruth([BASE_LINE], 'ground-truth-1', 'line-1', 'edited');
    expect(updated[0]?.line_transcriptions.find((tx) => tx.transcription_kind === 'ground_truth')?.text).toBe(
      'edited',
    );
    expect(withLocalGroundTruth([BASE_LINE], null, 'line-1', 'edited')).toEqual([BASE_LINE]);
  });
});
