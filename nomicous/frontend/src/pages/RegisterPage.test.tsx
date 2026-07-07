import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { RegisterPage } from './RegisterPage';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      register: vi.fn(),
    },
  };
});

function RequestedPage() {
  const location = useLocation();
  return <div>Requested Page editor{location.search}</div>;
}

describe('RegisterPage', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('creates an account and returns the user to the protected page they requested', async () => {
    vi.mocked(api.register).mockResolvedValue({
      access_token: 'new-user-token',
      token_type: 'bearer',
    });

    render(
      <MemoryRouter
        initialEntries={[
          {
            pathname: '/register',
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
          <Route path="/register" element={<RegisterPage />} />
          <Route
            path="/projects/:projectId/documents/:documentId/parts/:partId"
            element={<RequestedPage />}
          />
        </Routes>
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByLabelText('Email'), {
      target: { value: 'new.researcher@example.com' },
    });
    fireEvent.change(screen.getByLabelText('Username'), {
      target: { value: 'new-researcher' },
    });
    fireEvent.change(screen.getByLabelText(/^password/i), {
      target: { value: 'correct-password' },
    });
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));

    expect(await screen.findByText('Requested Page editor?panel=history')).toBeTruthy();
    await waitFor(() => {
      expect(localStorage.getItem('greekocr_access_token')).toBe('new-user-token');
    });
  });
});
