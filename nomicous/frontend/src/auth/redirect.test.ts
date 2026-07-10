import { describe, expect, it } from 'vitest';
import { authRedirectTarget } from './redirect';

describe('authRedirectTarget', () => {
  it('keeps only safe relative callback paths', () => {
    expect(authRedirectTarget('/projects/a?tab=jobs#recent')).toBe('/projects/a?tab=jobs#recent');
  });

  it.each([undefined, '', 'https://attacker.example', '//attacker.example', 'projects/a'])(
    'falls back to Projects for unsafe callback URL %s',
    (callbackUrl) => {
      expect(authRedirectTarget(callbackUrl)).toBe('/projects');
    },
  );
});
