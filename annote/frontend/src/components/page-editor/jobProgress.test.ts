import { describe, expect, it } from 'vitest';

import type { JobResponse } from '../../../api/client';
import { jobStatusLabel, transcribeLineProgress } from './jobProgress';

function job(partial: Partial<JobResponse>): JobResponse {
  return {
    id: 'job-1',
    type: 'transcribe',
    status: 'pending',
    payload: {},
    result: null,
    error: null,
    document_id: null,
    document_part_id: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    started_at: null,
    completed_at: null,
    ...partial,
  };
}

describe('jobProgress', () => {
  it('labels queued and waiting transcribe progress', () => {
    expect(jobStatusLabel(job({ status: 'pending' }))).toBe('Queued');
    expect(
      jobStatusLabel(
        job({
          status: 'waiting',
          payload: {
            ml_line_jobs: [{ ml_job_id: 'a', line_id: '1', line_index: 0 }],
            ml_line_outputs: {},
          },
        }),
      ),
    ).toBe('Transcribing 0/1 segment');
    expect(
      transcribeLineProgress(
        job({
          status: 'waiting',
          payload: {
            ml_line_jobs: [
              { ml_job_id: 'a', line_id: '1', line_index: 0 },
              { ml_job_id: 'b', line_id: '2', line_index: 1 },
            ],
            ml_line_outputs: { a: { text: 'hi', confidence: 1, character_confidences: [] } },
          },
        }),
      ),
    ).toBe('Transcribing 1/2 segments');
  });
});
