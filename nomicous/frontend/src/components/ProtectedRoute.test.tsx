import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ProtectedRoute } from './ProtectedRoute';

vi.mock('../auth/AuthProvider', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../auth/AuthProvider')>();
  return {
    ...actual,
    useAuthSession: vi.fn(),
  };
});

vi.mock('../auth/session', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../auth/session')>();
  return { ...actual, navigateToLogin: vi.fn() };
});

import { useAuthSession } from '../auth/AuthProvider';
import { navigateToLogin } from '../auth/session';

describe('ProtectedRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('redirects unauthenticated users to login', async () => {
    vi.mocked(useAuthSession).mockReturnValue({
      status: 'anonymous',
      establish: vi.fn(),
      logout: vi.fn(),
    });

    render(
      <ProtectedRoute>
        <div>Projects content</div>
      </ProtectedRoute>,
    );

    expect(screen.queryByText('Projects content')).toBeNull();
    await waitFor(() => {
      expect(navigateToLogin).toHaveBeenCalled();
    });
  });

  it('renders children for authenticated users', () => {
    vi.mocked(useAuthSession).mockReturnValue({
      status: 'authenticated',
      establish: vi.fn(),
      logout: vi.fn(),
    });

    render(
      <ProtectedRoute>
        <div>Projects content</div>
      </ProtectedRoute>,
    );

    expect(screen.getByText('Projects content')).toBeTruthy();
    expect(navigateToLogin).not.toHaveBeenCalled();
  });
});
