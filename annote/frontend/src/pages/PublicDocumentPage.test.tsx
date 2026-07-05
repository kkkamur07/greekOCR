import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  api,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
} from '../api/client';
import { PublicDocumentPage } from './PublicDocumentPage';

vi.mock('../components/ImageCanvas/ImageCanvas', () => ({
  default: ({
    regions,
    selectedRegionId,
    onSelectRegion,
  }: {
    regions: Array<{ id: number }>;
    selectedRegionId: number | null;
    onSelectRegion: (id: number | null) => void;
  }) => (
    <div data-testid="image-canvas">
      <span>Regions: {regions.length}</span>
      <span>Selected: {selectedRegionId ?? 'none'}</span>
      <button type="button" onClick={() => onSelectRegion(1)}>
        Select line 1
      </button>
    </div>
  ),
}));

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getPublicDocument: vi.fn(),
      getPublicLayout: vi.fn(),
      getPublicTranscriptionPdf: vi.fn(),
      getPublicPageXml: vi.fn(),
    },
  };
});

const DOCUMENT: DocumentWithPartsResponse = {
  id: 'doc-1',
  project_id: 'project-1',
  name: 'MS Or. 1445 — Genesis',
  workflow: 'published',
  created_at: '2026-06-16T10:00:00Z',
  updated_at: '2026-06-16T10:00:00Z',
  parts: [
    {
      id: 'part-1',
      document_id: 'doc-1',
      order: 0,
      image_url: '/public/media/parts/part-1',
      width: 640,
      height: 900,
      reviewed: true,
      created_at: '2026-06-16T10:00:00Z',
    },
  ],
};

const LAYOUT: PublicLayoutResponse = {
  blocks: [
    {
      id: 'block-1',
      part_id: 'part-1',
      order: 0,
      box: [0, 0, 640, 900],
    },
  ],
  lines: [
    {
      id: 'line-1',
      part_id: 'part-1',
      order: 0,
      points: [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ],
      line_transcriptions: [
        {
          id: 'lt-1',
          transcription_id: 'layer-1',
          transcription_kind: 'ground_truth',
          text: 'alpha beta',
          confidence: null,
        },
      ],
    },
  ],
};

function renderPublicPage() {
  return render(
    <MemoryRouter initialEntries={['/public/projects/project-1/documents/doc-1']}>
      <Routes>
        <Route
          path="/public/projects/:projectId/documents/:documentId"
          element={<PublicDocumentPage />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('PublicDocumentPage', () => {
  beforeEach(() => {
    vi.mocked(api.getPublicDocument).mockResolvedValue(DOCUMENT);
    vi.mocked(api.getPublicLayout).mockResolvedValue(LAYOUT);
    vi.mocked(api.getPublicTranscriptionPdf).mockResolvedValue(new Blob(['pdf'], { type: 'application/pdf' }));
    vi.mocked(api.getPublicPageXml).mockResolvedValue(new Blob(['xml'], { type: 'application/xml' }));
  });

  it('shows line geometry, transcription text, and download actions', async () => {
    renderPublicPage();

    expect(await screen.findByRole('heading', { name: 'MS Or. 1445 — Genesis' })).toBeInTheDocument();
    expect(screen.getByText(/1 page · Page 1: 1 line · 1 lines total/)).toBeInTheDocument();
    expect(screen.getByTestId('image-canvas')).toHaveTextContent('Regions: 1');
    expect(screen.getByText('alpha beta')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Preview PDF' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download PDF' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download PAGE XML' })).toBeInTheDocument();
  });

  it('syncs canvas selection with the transcript panel', async () => {
    renderPublicPage();
    await screen.findByRole('heading', { name: 'MS Or. 1445 — Genesis' });

    fireEvent.click(screen.getByRole('button', { name: 'Select line 1' }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Line 1' })).toBeInTheDocument();
    });
    expect(screen.getByTestId('image-canvas')).toHaveTextContent('Selected: 1');
  });
});
