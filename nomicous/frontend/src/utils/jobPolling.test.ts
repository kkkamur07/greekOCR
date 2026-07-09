import { afterEach, describe, expect, it, vi } from 'vitest';

import type { JobResponse } from '../api/client';
import { waitForJobViaSse, watchJobViaSse } from './jobPolling';

function job(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    id: 'job-1',
    type: 'pipeline',
    status: 'done',
    payload: {},
    result: { ok: true },
    error: null,
    user_id: 'user-1',
    document_id: null,
    document_part_id: null,
    created_at: '2026-07-09T10:00:00Z',
    updated_at: '2026-07-09T10:00:01Z',
    started_at: '2026-07-09T10:00:00Z',
    completed_at: '2026-07-09T10:00:01Z',
    ...overrides,
  };
}

function sseResponse(events: string[]): Response {
  const body = events.join('');
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

describe('watchJobViaSse', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('streams updates until the job reaches a terminal status', async () => {
    const running = job({ status: 'running' });
    const done = job();
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      sseResponse([
        `event: job\ndata: ${JSON.stringify(running)}\n\n`,
        `event: job\ndata: ${JSON.stringify(done)}\n\n`,
      ]),
    );
    const onUpdate = vi.fn();

    const cleanup = watchJobViaSse('job-1', {
      eventsUrl: 'http://localhost:8000/jobs/job-1/events',
      token: 'test-token',
      onUpdate,
    });

    await vi.waitFor(() => {
      expect(onUpdate).toHaveBeenCalledTimes(2);
    });
    expect(onUpdate).toHaveBeenNthCalledWith(1, running);
    expect(onUpdate).toHaveBeenNthCalledWith(2, done);
    cleanup();
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('falls back when the stream cannot be opened', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 503 }));
    const onUnavailable = vi.fn();

    watchJobViaSse('job-1', {
      eventsUrl: 'http://localhost:8000/jobs/job-1/events',
      onUpdate: vi.fn(),
      onUnavailable,
    });

    await vi.waitFor(() => {
      expect(onUnavailable).toHaveBeenCalledOnce();
    });
  });
});

describe('waitForJobViaSse', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('streams job updates with the bearer token header', async () => {
    const done = job();
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        sseResponse([`event: job\ndata: ${JSON.stringify(done)}\n\n`]),
      );
    const onUpdate = vi.fn();

    const result = await waitForJobViaSse('job-1', {
      eventsUrl: 'http://localhost:8000/jobs/job-1/events',
      token: 'test-token',
      onUpdate,
    });

    expect(result).toEqual(done);
    expect(onUpdate).toHaveBeenCalledWith(done);
    const [, init] = fetchMock.mock.calls[0];
    const headers = init?.headers as Headers;
    expect(headers.get('Accept')).toBe('text/event-stream');
    expect(headers.get('Authorization')).toBe('Bearer test-token');
  });

  it('rejects failed job events with the server error message', async () => {
    const failed = job({
      status: 'failed',
      error: 'intentional failure',
      result: null,
    });
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      sseResponse([`event: job\ndata: ${JSON.stringify(failed)}\n\n`]),
    );

    await expect(
      waitForJobViaSse('job-1', {
        eventsUrl: 'http://localhost:8000/jobs/job-1/events',
      }),
    ).rejects.toThrow('intentional failure');
  });
});
