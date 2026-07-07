import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import {
  PageEditorProcessingBanner,
  getPageEditorProcessingLabel,
} from './PageEditorProcessingBanner';

describe('PageEditorProcessingBanner', () => {
  it('renders nothing when idle', () => {
    const { container } = render(<PageEditorProcessingBanner kind={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows segmentation in progress', () => {
    render(<PageEditorProcessingBanner kind="segmentation" />);
    expect(screen.getByRole('status')).toHaveTextContent('Segmentation in progress');
  });

  it('shows transcription labels by scope', () => {
    expect(getPageEditorProcessingLabel('transcription-page')).toBe(
      'Full-page transcription in progress',
    );
    expect(getPageEditorProcessingLabel('transcription-segment')).toBe(
      'Segment transcription in progress',
    );
  });
});
