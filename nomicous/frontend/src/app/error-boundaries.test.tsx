import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import GlobalError from './global-error.next';
import PartEditorError from './projects/[projectId]/documents/[documentId]/parts/[partId]/error.next';

describe('Next error boundaries', () => {
  it('renders a recoverable root error', () => {
    const reset = vi.fn();

    render(<GlobalError error={new Error('boom')} reset={reset} />);
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));

    expect(screen.getByRole('alert')).toHaveTextContent('Something went wrong');
    expect(reset).toHaveBeenCalledOnce();
  });

  it('renders a recoverable part editor error', () => {
    const reset = vi.fn();

    render(<PartEditorError error={new Error('boom')} reset={reset} />);
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));

    expect(screen.getByRole('alert')).toHaveTextContent('Unable to open the page editor');
    expect(reset).toHaveBeenCalledOnce();
  });
});
