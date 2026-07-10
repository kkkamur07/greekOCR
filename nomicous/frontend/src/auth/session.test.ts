import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  clearLoginRedirectGuard,
  navigateToLogin,
  redirectToLogin,
} from './session';

describe('login redirects', () => {
  beforeEach(() => {
    clearLoginRedirectGuard();
    window.history.replaceState({}, '', '/projects/project-1');
  });

  afterEach(() => {
    clearLoginRedirectGuard();
  });

  it('allows only one redirect across API and React callers', () => {
    const router = { replace: vi.fn() };

    navigateToLogin(router);
    redirectToLogin();

    expect(router.replace).toHaveBeenCalledWith(
      '/login?callbackUrl=%2Fprojects%2Fproject-1',
    );
    expect(router.replace).toHaveBeenCalledOnce();
  });
});
