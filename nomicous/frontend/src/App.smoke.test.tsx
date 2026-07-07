import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import App from './App';

vi.mock('./auth/storage', () => ({
  getAccessToken: () => null,
  setAccessToken: vi.fn(),
  clearAccessToken: vi.fn(),
}));

describe('App smoke', () => {
  it('renders the login page for unauthenticated users', async () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    );

    expect(await screen.findByRole('heading', { name: /sign in/i })).toBeTruthy();
  });
});
