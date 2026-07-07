import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../../api/client';
import { ApiError } from '../../api/errors';
import { PageEditorTranscriptionPdfPane } from './PageEditorTranscriptionPdfPane';

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      generateTranscriptionPdf: vi.fn(),
    },
  };
});

const mockedGenerateTranscriptionPdf = api.generateTranscriptionPdf as ReturnType<typeof vi.fn>;

describe('PageEditorTranscriptionPdfPane', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGenerateTranscriptionPdf.mockResolvedValue(
      new Blob(['%PDF'], { type: 'application/pdf' }),
    );
  });

  it('loads the transcription PDF into an embedded preview', async () => {
    const createObjectURL = vi.fn(() => 'blob:preview');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTitle('Transcription PDF preview')).toHaveAttribute('data', 'blob:preview');
    });
    expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledWith('project-1', 'doc-1', 'part-1');

    vi.unstubAllGlobals();
  });

  it('refetches when refreshKey changes', async () => {
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:preview'),
      revokeObjectURL: vi.fn(),
    });

    const { rerender } = render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledTimes(1);
    });

    rerender(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={2}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledTimes(2);
    });

    vi.unstubAllGlobals();
  });

  it('downloads the PDF when the user clicks Download', async () => {
    const blob = new Blob(['%PDF'], { type: 'application/pdf' });
    mockedGenerateTranscriptionPdf.mockResolvedValue(blob);
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:download'),
      revokeObjectURL: vi.fn(),
    });

    const anchor = { click: vi.fn(), href: '', download: '' } as unknown as HTMLAnchorElement;
    const originalCreateElement = document.createElement.bind(document);
    const createElement = vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => {
      if (tagName === 'a') return anchor;
      return originalCreateElement(tagName, options);
    });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByRole('button', { name: /download pdf/i }));

    await waitFor(() => {
      expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledTimes(2);
      expect(anchor.download).toBe('page-1_transcription.pdf');
      expect(anchor.click).toHaveBeenCalled();
    });

    createElement.mockRestore();
    vi.unstubAllGlobals();
  });

  it('shows API errors from transcription PDF generation', async () => {
    mockedGenerateTranscriptionPdf.mockRejectedValue(new ApiError('Forbidden', 403));
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(),
      revokeObjectURL: vi.fn(),
    });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText('Forbidden')).toBeTruthy();

    vi.unstubAllGlobals();
  });
});
