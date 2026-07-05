import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../../api/client';
import { PageEditorSharingMenu } from './PageEditorSharingMenu';

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      updateDocument: vi.fn(),
    },
  };
});

const mockedUpdateDocument = api.updateDocument as ReturnType<typeof vi.fn>;

function renderMenu(workflow: 'draft' | 'published' | 'archived' = 'draft') {
  const onWorkflowChange = vi.fn();
  render(
    <MemoryRouter>
      <div role="menu">
        <PageEditorSharingMenu
          projectId="project-1"
          documentId="doc-1"
          workflow={workflow}
          onWorkflowChange={onWorkflowChange}
        />
      </div>
    </MemoryRouter>,
  );
  return { onWorkflowChange };
}

describe('PageEditorSharingMenu', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedUpdateDocument.mockResolvedValue({
      id: 'doc-1',
      project_id: 'project-1',
      name: 'Grec 1360',
      workflow: 'published',
      created_at: '2026-06-16T10:00:00Z',
      updated_at: '2026-06-16T10:00:00Z',
      part_count: 1,
    });
  });

  it('publishes a draft document from the sharing section', async () => {
    const { onWorkflowChange } = renderMenu('draft');

    fireEvent.click(screen.getByRole('menuitem', { name: /publish live page/i }));

    await waitFor(() => {
      expect(mockedUpdateDocument).toHaveBeenCalledWith('project-1', 'doc-1', {
        workflow: 'published',
      });
      expect(onWorkflowChange).toHaveBeenCalledWith('published');
    });
  });

  it('shows the public link when the document is published', () => {
    renderMenu('published');

    expect(screen.getByLabelText(/public document url/i)).toHaveValue(
      `${window.location.origin}/public/projects/project-1/documents/doc-1`,
    );
    expect(screen.getByRole('link', { name: /open public view/i })).toHaveAttribute(
      'href',
      '/public/projects/project-1/documents/doc-1',
    );
  });
});
