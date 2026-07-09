import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ProtectedRoute } from './ProtectedRoute';

const navigateMock = vi.fn();

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

vi.mock('../auth/session', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../auth/session')>();
  return {
    ...actual,
    hasAccessToken: vi.fn(),
    navigateToLogin: vi.fn(),
  };
});

import { hasAccessToken, navigateToLogin } from '../auth/session';

describe('ProtectedRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('redirects unauthenticated users to login', async () => {
    vi.mocked(hasAccessToken).mockReturnValue(false);

    render(
      <MemoryRouter initialEntries={['/projects']}>
        <Routes>
          <Route
            path="/projects"
            element={
              <ProtectedRoute>
                <div>Projects content</div>
              </ProtectedRoute>
            }
          />
          <Route path="/login" element={<div>Login page</div>} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByText('Projects content')).toBeNull();
    await waitFor(() => {
      expect(navigateToLogin).toHaveBeenCalled();
    });
  });

  it('renders children for authenticated users', () => {
    vi.mocked(hasAccessToken).mockReturnValue(true);

    render(
      <MemoryRouter initialEntries={['/projects']}>
        <Routes>
          <Route
            path="/projects"
            element={
              <ProtectedRoute>
                <div>Projects content</div>
              </ProtectedRoute>
            }
          />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('Projects content')).toBeTruthy();
    expect(navigateToLogin).not.toHaveBeenCalled();
  });
});
