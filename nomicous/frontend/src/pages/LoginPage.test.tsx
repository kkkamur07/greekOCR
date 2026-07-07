import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { LoginPage } from './LoginPage';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      login: vi.fn(),
    },
  };
});

function RequestedPage() {
  const location = useLocation();
  return <div>Requested Page editor{location.search}</div>;
}

describe('LoginPage', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('signs in and returns the user to the protected page they requested', async () => {
    vi.mocked(api.login).mockResolvedValue({
      access_token: 'jwt-token',
      token_type: 'bearer',
    });

    render(
      <MemoryRouter
        initialEntries={[
          {
            pathname: '/login',
            state: {
              from: {
                pathname: '/projects/project-1/documents/doc-1/parts/part-1',
                search: '?panel=history',
              },
            },
          },
        ]}
      >
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/projects/:projectId/documents/:documentId/parts/:partId"
            element={<RequestedPage />}
          />
        </Routes>
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByLabelText('Email'), {
      target: { value: 'researcher@example.com' },
    });
    fireEvent.change(screen.getByLabelText('Password'), {
      target: { value: 'correct-password' },
    });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    expect(await screen.findByText('Requested Page editor?panel=history')).toBeTruthy();
    await waitFor(() => {
      expect(localStorage.getItem('greekocr_access_token')).toBe('jwt-token');
    });
  });
});
