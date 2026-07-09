import { describe, expect, it, vi } from 'vitest';
import { dedupedGet } from './getCache';

describe('dedupedGet', () => {
  it('shares one in-flight request for the same key', async () => {
    const fetcher = vi.fn(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return 'ok';
    });

    const [first, second] = await Promise.all([
      dedupedGet('GET /example', fetcher),
      dedupedGet('GET /example', fetcher),
    ]);

    expect(first).toBe('ok');
    expect(second).toBe('ok');
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('allows a new request after the previous one settles', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce('first')
      .mockResolvedValueOnce('second');

    expect(await dedupedGet('GET /example', fetcher)).toBe('first');
    expect(await dedupedGet('GET /example', fetcher)).toBe('second');
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
