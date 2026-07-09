import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '../../api/errors';
import {
  DOCUMENT,
  enableBaselinesOnCanvas,
  flushPageEditorEffects,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from './testSupport';

describe('PageEditorPlaceholderPage shell', () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it('opens an authenticated document part in the annote page workspace', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    expect(screen.getByText('Grec 1360 · Page 1')).toBeTruthy();
    expect(screen.getByAltText('Page 1')).toBeTruthy();
    expect(screen.getByRole('link', { name: /back to document/i }).getAttribute('href')).toBe(
      '/projects/project-1/documents/doc-1',
    );
  });

  it('renders classic strip layout with toolbar, canvas, and pairing strip when a Segment is selected', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    expect(screen.getByRole('button', { name: /rectangle segment/i })).toBeTruthy();
    expect(screen.getByRole('img', { name: /page geometry canvas/i })).toBeTruthy();

    fireEvent.click(screen.getByLabelText(/^Segment 1/));

    expect(screen.getByRole('heading', { name: /Segment 1/i })).toBeTruthy();
    expect(screen.getByText('Segment 1')).toBeTruthy();
    expect(screen.getByRole('progressbar', { name: /pairing progress/i })).toBeTruthy();
  });

  it('does not render protected media when the API rejects access', async () => {
    mockedApi.getDocument.mockRejectedValue(new ApiError('Forbidden', 403));

    renderPageEditor();

    expect(await screen.findByText('Page unavailable')).toBeTruthy();
    expect(screen.getByText('This page is not available to your account.')).toBeTruthy();
    expect(screen.queryByAltText('Page 1')).toBeNull();
  });
  it('opens the transcription PDF side pane from the toolbar', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.generateTranscriptionPdf.mockResolvedValue(
      new Blob(['%PDF'], { type: 'application/pdf' }),
    );
    const originalUrl = globalThis.URL;
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:preview'),
      revokeObjectURL: vi.fn(),
    });

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /toggle transcription pdf/i }));

    expect(await screen.findByLabelText('Transcription PDF preview')).toBeTruthy();
    await waitFor(() => {
      expect(mockedApi.generateTranscriptionPdf).toHaveBeenCalledWith(
        'project-1',
        'doc-1',
        'part-1',
      );
    });

    vi.stubGlobal('URL', originalUrl);
  });

  it('publishes the document from the Process sharing section', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    fireEvent.click(await screen.findByRole('button', { name: /^process/i }));
    fireEvent.click(await screen.findByRole('menuitem', { name: /publish live page/i }));

    await waitFor(() => {
      expect(mockedApi.updateDocument).toHaveBeenCalledWith('project-1', 'doc-1', {
        workflow: 'published',
      });
    });
    expect(screen.getByLabelText(/public document url/i)).toBeTruthy();
  });
});
