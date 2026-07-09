import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { ApiError } from '../api/errors';
import * as session from '../auth/session';
import { ProjectsPage } from './ProjectsPage';

const navigateMock = vi.fn();

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return {
    ...actual,
    useNavigate: () => navigateMock,
    useLocation: () => ({ pathname: '/projects', search: '', hash: '' }),
  };
});

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      me: vi.fn(),
      listProjects: vi.fn(),
      deleteProject: vi.fn(),
      updateProject: vi.fn(),
    },
  };
});

describe('ProjectsPage', () => {
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
    vi.mocked(api.listProjects).mockResolvedValue([
      {
        id: 'project-1',
        name: 'Syriac Pentateuch',
        slug: 'syriac-pentateuch',
        guidelines: null,
        owner_id: 'user-1',
        document_count: 2,
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-01-01T00:00:00Z',
      },
    ]);
  });

  it('lets the owner delete an owned project from the table', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.mocked(api.deleteProject).mockResolvedValue(undefined);

    render(
      <MemoryRouter>
        <ProjectsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('heading', { name: 'Projects' });
    fireEvent.click(screen.getByRole('button', { name: /delete project syriac pentateuch/i }));

    await waitFor(() => {
      expect(api.deleteProject).toHaveBeenCalledWith('project-1');
    });
  });

  it('redirects to login when the session is unauthorized', async () => {
    vi.mocked(api.me).mockRejectedValue(new ApiError('Unauthorized', 401));

    render(
      <MemoryRouter>
        <ProjectsPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(screen.queryByText('Projects unavailable')).toBeNull();
  });

  it('redirects to login when no access token is present', async () => {
    vi.spyOn(session, 'hasAccessToken').mockReturnValue(false);

    render(
      <MemoryRouter>
        <ProjectsPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(api.me).not.toHaveBeenCalled();
  });
});
