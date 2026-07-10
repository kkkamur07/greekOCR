import { describe, expect, it } from 'vitest';

import { collectCursorPages } from './cursorPagination';

describe('collectCursorPages', () => {
  it('rejects a repeated cursor instead of fetching indefinitely', async () => {
    await expect(
      collectCursorPages(
        async ({ cursor }) => ({
          items: [cursor ?? 'first'],
          next_cursor: 'repeated-cursor',
        }),
        { maxPages: 5 },
      ),
    ).rejects.toThrow('repeated a cursor');
  });

  it('stops at the configured maximum page count', async () => {
    await expect(
      collectCursorPages(
        async ({ cursor }) => ({
          items: [cursor ?? 'first'],
          next_cursor: `${cursor ?? 'next'}-next`,
        }),
        { maxPages: 2 },
      ),
    ).rejects.toThrow('2-page limit');
  });

  it('honours cancellation before starting another page', async () => {
    const controller = new AbortController();
    await expect(
      collectCursorPages(
        async () => {
          controller.abort();
          return { items: ['first'], next_cursor: 'next' };
        },
        { signal: controller.signal },
      ),
    ).rejects.toMatchObject({ name: 'AbortError' });
  });
});
