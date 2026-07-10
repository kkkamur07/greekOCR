import { describe, expect, it, vi } from 'vitest';
import {
  dedupedGet,
  getAuthRequestVersion,
  invalidateAuthGetCache,
} from './getCache';

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

  it('does not share in-flight GETs across auth-session changes', async () => {
    let releaseFirst: (value: string) => void;
    const first = dedupedGet('GET /projects', () => new Promise<string>((resolve) => {
      releaseFirst = resolve;
    }));

    const before = getAuthRequestVersion();
    invalidateAuthGetCache();
    expect(getAuthRequestVersion()).toBe(before + 1);

    const secondFetcher = vi.fn().mockResolvedValue('second-session');
    const second = dedupedGet('GET /projects', secondFetcher);
    releaseFirst!('first-session');

    await expect(first).resolves.toBe('first-session');
    await expect(second).resolves.toBe('second-session');
    expect(secondFetcher).toHaveBeenCalledTimes(1);
  });
});
