import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api, type DocumentWithPartsResponse } from '../api/client';
import { DocumentDetailPage } from './DocumentDetailPage';

vi.mock('../components/AuthenticatedImage', () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock('../components/JobsPanel/JobsPanel', () => ({
  JobsPanel: () => <div>Jobs panel</div>,
}));

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getDocument: vi.fn(),
      uploadPart: vi.fn(),
      reorderParts: vi.fn(),
      deletePart: vi.fn(),
    },
  };
});

const DOCUMENT: DocumentWithPartsResponse = {
  id: 'doc-1',
  project_id: 'project-1',
  name: 'Grec 1360',
  workflow: 'draft',
  created_at: '2026-06-16T10:00:00Z',
  updated_at: '2026-06-16T10:00:00Z',
  parts: [
    {
      id: 'part-2',
      document_id: 'doc-1',
      order: 1,
      image_url: '/media/parts/part-2',
      width: 800,
      height: 1000,
      created_at: '2026-06-16T10:00:00Z',
    },
    {
      id: 'part-1',
      document_id: 'doc-1',
      order: 0,
      image_url: '/media/parts/part-1',
      width: 640,
      height: 900,
      created_at: '2026-06-16T10:00:00Z',
    },
  ],
};

function renderDocumentPage() {
  return render(
    <MemoryRouter initialEntries={['/projects/project-1/documents/doc-1']}>
      <Routes>
        <Route
          path="/projects/:projectId/documents/:documentId"
          element={<DocumentDetailPage />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('DocumentDetailPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getDocument).mockResolvedValue(DOCUMENT);
  });

  it('lists document parts in order and links each part to the protected page route', async () => {
    renderDocumentPage();

    await screen.findByText('Grec 1360');

    const pageLinks = screen.getAllByRole('link', { name: /open page/i });
    expect(pageLinks).toHaveLength(2);
    expect(pageLinks[0].getAttribute('href')).toBe(
      '/projects/project-1/documents/doc-1/parts/part-1',
    );
    expect(pageLinks[1].getAttribute('href')).toBe(
      '/projects/project-1/documents/doc-1/parts/part-2',
    );

    await waitFor(() => {
      expect(screen.getByAltText('Part 1')).toBeTruthy();
      expect(screen.getByAltText('Part 2')).toBeTruthy();
    });
  });
});
