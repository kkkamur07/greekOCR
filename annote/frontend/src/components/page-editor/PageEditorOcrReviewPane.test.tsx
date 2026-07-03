import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { PageEditorOcrReviewPane } from './PageEditorOcrReviewPane';

describe('PageEditorOcrReviewPane', () => {
  it('renders line confidence and highlighted OCR text for a model transcription', () => {
    render(
      <PageEditorOcrReviewPane
        segmentNumber={2}
        transcription={{
          id: 'tx-1',
          transcription_id: 'model-1',
          transcription_kind: 'model',
          text: 'ab',
          confidence: 0.91,
          character_confidences: [
            { char: 'a', confidence: 0.95 },
            { char: 'b', confidence: 0.62 },
          ],
        }}
      />,
    );

    expect(screen.getByText('OCR review · Segment 2')).toBeTruthy();
    expect(screen.getByLabelText('Line confidence')).toBeTruthy();
    expect(screen.getByText('91.0%')).toBeTruthy();
    expect(screen.getByLabelText('OCR text with per-character confidence highlighting')).toBeTruthy();
    expect(screen.getByTitle('a: 95.0%')).toBeTruthy();
    expect(screen.getByTitle('b: 62.0%')).toBeTruthy();
  });

  it('shows a placeholder when no transcription is available', () => {
    render(<PageEditorOcrReviewPane segmentNumber={1} transcription={null} />);

    expect(screen.getByText('Segment 1 has no OCR text on this layer yet.')).toBeTruthy();
  });
});
