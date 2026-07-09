import type { Location, NavigateFunction } from 'react-router-dom';

import { ApiError } from '../api/errors';
import { clearAccessToken, getAccessToken } from './storage';

export function hasAccessToken(): boolean {
  const token = getAccessToken();
  return typeof token === 'string' && token.trim().length > 0;
}

/** Full-page redirect for API-layer auth failures outside React Router. */
export function redirectToLogin(): void {
  clearAccessToken();
  const loginPath = '/login';
  if (window.location.pathname !== loginPath && window.location.pathname !== '/register') {
    window.location.assign(loginPath);
  }
}

export function navigateToLogin(
  navigate: NavigateFunction,
  location: Pick<Location, 'pathname' | 'search' | 'hash'>,
): void {
  clearAccessToken();
  navigate('/login', { replace: true, state: { from: location } });
}

export function isUnauthorized(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401;
}
