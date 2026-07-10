import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { testRouter } from '../../vitest.setup';

import { api, type DocumentWithPartsResponse } from '../api/client';
import { ApiError } from '../api/errors';
import * as session from '../auth/session';
import { DocumentDetailPage } from './DocumentDetailPage';

vi.mock('../components/AuthenticatedImage', () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock('../components/document/JobsNotice', () => ({
  JobsNotice: () => <div>Jobs panel</div>,
}));

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getDocument: vi.fn(),
      me: vi.fn(),
      getProject: vi.fn(),
      uploadPart: vi.fn(),
      reorderParts: vi.fn(),
      deletePart: vi.fn(),
      updatePartReviewStatus: vi.fn(),
      updateDocument: vi.fn(),
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
  part_count: 2,
  parts: [
    {
      id: 'part-2',
      document_id: 'doc-1',
      order: 1,
      image_url: '/media/parts/part-2',
      width: 800,
      height: 1000,
      reviewed: false,
      created_at: '2026-06-16T10:00:00Z',
    },
    {
      id: 'part-1',
      document_id: 'doc-1',
      order: 0,
      image_url: '/media/parts/part-1',
      width: 640,
      height: 900,
      reviewed: false,
      created_at: '2026-06-16T10:00:00Z',
    },
  ],
};

function renderDocumentPage(initialPath = '/projects/project-1/documents/doc-1') {
  window.history.replaceState({}, '', initialPath);
  return render(<DocumentDetailPage />);
}

describe('DocumentDetailPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(session, 'hasAccessToken').mockReturnValue(true);
    vi.spyOn(session, 'navigateToLogin').mockImplementation(() => {});
    vi.mocked(api.me).mockResolvedValue({
      id: 'user-1',
      email: 'dev@example.com',
      username: 'dev',
      created_at: '2026-01-01T00:00:00Z',
    });
    vi.mocked(api.getProject).mockResolvedValue({
      id: 'project-1',
      name: 'Test Project',
      slug: 'test-project',
      owner_id: 'user-1',
      guidelines: null,
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
      document_count: 1,
    });
    vi.mocked(api.getDocument).mockResolvedValue(DOCUMENT);
  });

  it('lists document parts in order and opens the page editor when a row is clicked', async () => {
    renderDocumentPage();

    await screen.findByRole('heading', { name: 'Grec 1360' });

    expect(screen.getByAltText('Part 1')).toBeTruthy();
    expect(screen.getByAltText('Part 2')).toBeTruthy();

    const rows = screen.getAllByRole('listitem');
    expect(rows).toHaveLength(2);

    fireEvent.click(rows[0]);
    await waitFor(() => {
      expect(testRouter().push).toHaveBeenCalledWith('/projects/project-1/documents/doc-1/parts/part-1');
    });
  });

  it('shows review status on each part and lets a project member mark a part reviewed', async () => {
    vi.mocked(api.updatePartReviewStatus).mockResolvedValue({
      ...DOCUMENT.parts[1],
      reviewed: true,
    });

    renderDocumentPage();

    await screen.findByRole('heading', { name: 'Grec 1360' });
    expect(screen.getAllByText('unreviewed')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /mark part 1 reviewed/i }));

    await waitFor(() => {
      expect(api.updatePartReviewStatus).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { reviewed: true },
      );
    });
  });

  it('lets a project member mark a reviewed part unreviewed', async () => {
    vi.mocked(api.getDocument).mockResolvedValue({
      ...DOCUMENT,
      parts: [
        { ...DOCUMENT.parts[1], reviewed: true },
        DOCUMENT.parts[0],
      ],
    });
    vi.mocked(api.updatePartReviewStatus).mockResolvedValue({
      ...DOCUMENT.parts[0],
      reviewed: false,
    });

    renderDocumentPage();

    await screen.findByRole('heading', { name: 'Grec 1360' });
    expect(screen.getByText('reviewed')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: /mark part 1 unreviewed/i }));

    await waitFor(() => {
      expect(api.updatePartReviewStatus).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { reviewed: false },
      );
    });
  });

  it('keeps review status when the API rejects the change', async () => {
    vi.mocked(api.updatePartReviewStatus).mockRejectedValue(new ApiError('Forbidden', 403));

    renderDocumentPage();

    await screen.findByRole('heading', { name: 'Grec 1360' });
    fireEvent.click(screen.getByRole('button', { name: /mark part 1 reviewed/i }));

    await waitFor(() => {
      expect(api.updatePartReviewStatus).toHaveBeenCalled();
    });
    expect(screen.getAllByText('unreviewed')).toHaveLength(2);
  });

  it('opens live sharing from the document header and publishes the document', async () => {
    vi.mocked(api.updateDocument).mockResolvedValue({
      id: 'doc-1',
      project_id: 'project-1',
      name: 'Grec 1360',
      workflow: 'published',
      part_count: 2,
      created_at: '2026-06-16T10:00:00Z',
      updated_at: '2026-06-16T10:00:00Z',
    });

    renderDocumentPage();

    fireEvent.click(await screen.findByRole('button', { name: /grec 1360, click to edit/i }));
    fireEvent.click(screen.getByRole('button', { name: /publish live page/i }));

    await waitFor(() => {
      expect(api.updateDocument).toHaveBeenCalledWith('project-1', 'doc-1', {
        workflow: 'published',
      });
    });
    expect(screen.getByLabelText(/public document url/i)).toBeTruthy();
  });

  it('renames the document from the header panel', async () => {
    vi.mocked(api.updateDocument).mockResolvedValue({
      id: 'doc-1',
      project_id: 'project-1',
      name: 'MS Or. 1445 — Genesis',
      workflow: 'draft',
      part_count: 2,
      created_at: '2026-06-16T10:00:00Z',
      updated_at: '2026-06-16T10:00:00Z',
    });

    renderDocumentPage();

    fireEvent.click(await screen.findByRole('button', { name: /grec 1360, click to edit/i }));
    fireEvent.change(screen.getByLabelText('Name'), {
      target: { value: 'MS Or. 1445 — Genesis' },
    });
    fireEvent.click(screen.getByRole('button', { name: /save name/i }));

    await waitFor(() => {
      expect(api.updateDocument).toHaveBeenCalledWith('project-1', 'doc-1', {
        name: 'MS Or. 1445 — Genesis',
      });
    });
    expect(screen.getByRole('heading', { name: 'MS Or. 1445 — Genesis' })).toBeTruthy();
  });

  it('redirects to login when the session is unauthorized', async () => {
    vi.mocked(api.getDocument).mockRejectedValue(new ApiError('Unauthorized', 401));

    renderDocumentPage();

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(screen.queryByText('This document is not available to your account.')).toBeNull();
  });

  it('redirects to login when no access token is present', async () => {
    vi.spyOn(session, 'hasAccessToken').mockReturnValue(false);

    renderDocumentPage();

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(api.getDocument).not.toHaveBeenCalled();
  });
});
