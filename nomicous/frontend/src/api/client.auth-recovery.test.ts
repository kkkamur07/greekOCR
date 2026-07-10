import { beforeEach, describe, expect, it, vi } from 'vitest';

const { redirectToLogin } = vi.hoisted(() => ({
  redirectToLogin: vi.fn(),
}));

vi.mock('../auth/session', () => ({
  redirectToLogin,
}));

import { apiRequest, fetchBinaryApi } from './client';
import { ApiError } from './errors';
import { clearAccessToken, getAccessToken, setAccessToken } from '../auth/storage';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('API auth recovery', () => {
  beforeEach(() => {
    clearAccessToken();
    redirectToLogin.mockReset();
  });

  it('refreshes and retries a protected JSON request once', async () => {
    setAccessToken('expired-token');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(jsonResponse({ access_token: 'fresh-token' }))
      .mockResolvedValueOnce(jsonResponse({ id: 'project-1' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiRequest<{ id: string }>('/projects/project-1')).resolves.toEqual({
      id: 'project-1',
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get('Authorization')).toBe(
      'Bearer expired-token',
    );
    expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get('Authorization')).toBe(
      'Bearer fresh-token',
    );
    expect(getAccessToken()).toBe('fresh-token');
  });

  it('shares one refresh across concurrent protected requests', async () => {
    setAccessToken('expired-token');
    let resolveRefresh: ((response: Response) => void) | undefined;
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith('/auth/refresh')) {
        return new Promise<Response>((resolve) => {
          resolveRefresh = resolve;
        });
      }
      const token = new Headers(init?.headers).get('Authorization');
      return Promise.resolve(
        token === 'Bearer fresh-token'
          ? jsonResponse({ path: url })
          : new Response(null, { status: 401 }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const first = apiRequest<{ path: string }>('/projects/one');
    const second = apiRequest<{ path: string }>('/projects/two');

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    resolveRefresh?.(jsonResponse({ access_token: 'fresh-token' }));

    await expect(Promise.all([first, second])).resolves.toEqual([
      { path: expect.stringContaining('/projects/one') },
      { path: expect.stringContaining('/projects/two') },
    ]);
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith('/auth/refresh'))).toHaveLength(1);
  });

  it('retries protected binary requests with the refreshed token', async () => {
    setAccessToken('expired-token');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(jsonResponse({ access_token: 'fresh-token' }))
      .mockResolvedValueOnce(new Response('pdf-data', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await fetchBinaryApi('/documents/one/download');
    await expect(result.text()).resolves.toBe('pdf-data');
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get('Authorization')).toBe(
      'Bearer fresh-token',
    );
  });

  it('does not refresh skipAuth routes', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiRequest('/public/projects/one', { skipAuth: true })).rejects.toBeInstanceOf(
      ApiError,
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(redirectToLogin).not.toHaveBeenCalled();
  });

  it('keeps abortable GET requests independent', async () => {
    const pending: Array<(response: Response) => void> = [];
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise<Response>((resolve, reject) => {
        pending.push(resolve);
        init?.signal?.addEventListener(
          'abort',
          () => reject(init.signal?.reason ?? new DOMException('Aborted', 'AbortError')),
          { once: true },
        );
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const firstController = new AbortController();
    const secondController = new AbortController();
    const first = apiRequest<{ source: string }>('/projects/project-1/jobs?limit=8', {
      signal: firstController.signal,
    });
    const second = apiRequest<{ source: string }>('/projects/project-1/jobs?limit=8', {
      signal: secondController.signal,
    });

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    firstController.abort();
    await expect(first).rejects.toMatchObject({ name: 'AbortError' });

    pending[1](jsonResponse({ source: 'second request' }));
    await expect(second).resolves.toEqual({ source: 'second request' });
  });

  it('stops after a second 401 and redirects once', async () => {
    setAccessToken('expired-token');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(jsonResponse({ access_token: 'fresh-token' }))
      .mockResolvedValueOnce(new Response(null, { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiRequest('/projects/one')).rejects.toMatchObject({ status: 401 });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(redirectToLogin).toHaveBeenCalledOnce();
  });
});
