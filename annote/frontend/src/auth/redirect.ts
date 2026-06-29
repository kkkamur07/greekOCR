export function authRedirectTarget(state: unknown): string {
  if (
    typeof state !== 'object' ||
    state === null ||
    !('from' in state) ||
    typeof state.from !== 'object' ||
    state.from === null ||
    !('pathname' in state.from) ||
    typeof state.from.pathname !== 'string'
  ) {
    return '/projects';
  }

  const search =
    'search' in state.from && typeof state.from.search === 'string' ? state.from.search : '';
  const hash = 'hash' in state.from && typeof state.from.hash === 'string' ? state.from.hash : '';
  return `${state.from.pathname}${search}${hash}`;
}
