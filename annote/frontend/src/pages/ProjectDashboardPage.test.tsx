import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { ApiError } from '../api/errors';
import { ProjectDashboardPage } from './ProjectDashboardPage';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getProject: vi.fn(),
      listDocuments: vi.fn(),
      createDocument: vi.fn(),
    },
  };
});

function renderProjectDashboard() {
  return render(
    <MemoryRouter initialEntries={['/projects/project-1']}>
      <Routes>
        <Route path="/projects/:projectId" element={<ProjectDashboardPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ProjectDashboardPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows an unavailable state instead of document actions when project access is rejected', async () => {
    vi.mocked(api.getProject).mockRejectedValue(new ApiError('Forbidden', 403));
    vi.mocked(api.listDocuments).mockResolvedValue([]);

    renderProjectDashboard();

    expect(await screen.findByText('Project unavailable')).toBeTruthy();
    expect(screen.getByText('This project is not available to your account.')).toBeTruthy();
    expect(screen.queryByRole('button', { name: /new document/i })).toBeNull();
  });
});
