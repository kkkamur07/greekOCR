/**
 * The consent token arrives in the URL fragment, and has to leave it again.
 *
 * A fragment is never sent to the server: it stays out of access logs, out of
 * `Referer`, and out of the RSC requests the App Router makes on navigation.
 * That property is only worth having if the token never reaches a query string
 * either - and it would, unprompted: an unauthenticated researcher opening the
 * link is bounced through `navigateToLogin`, which folds
 * `window.location.hash` into the `callbackUrl` *query* parameter of `/login`.
 * The token would arrive at the server on the very redirect meant to protect
 * the page.
 *
 * So the fragment is taken out of the address bar as soon as the route renders,
 * before the session has finished restoring, and parked in `sessionStorage` -
 * per tab, same origin, never transmitted. It is dropped again the moment the
 * pairing it names is approved, denied, or found to be no longer valid.
 */
const STORAGE_KEY = "nomicous.pairing-verification-token";

/**
 * `sessionStorage` access throws outright in some blocked-cookie
 * configurations, so every use goes through here. Losing the parking space is
 * survivable - the token is still returned to the caller that took it from the
 * URL, which is enough to finish pairing in a tab that never leaves the page.
 */
function sessionStore(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function clean(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

/**
 * Take the token out of the address bar, park it, and hand it back.
 *
 * Returns `null` when the fragment is empty, which is the normal case on every
 * render after the first and on the way back from a login round trip. Safe to
 * call repeatedly, and safe to call from two components in the same commit.
 */
export function takePairingTokenFromUrl(): string | null {
  if (typeof window === "undefined") return null;
  const token = clean(window.location.hash.replace(/^#/, ""));
  if (!token) return null;
  sessionStore()?.setItem(STORAGE_KEY, token);
  window.history.replaceState(
    window.history.state,
    "",
    `${window.location.pathname}${window.location.search}`,
  );
  return token;
}

/** The parked token, if this tab still holds one. */
export function stashedPairingToken(): string | null {
  return clean(sessionStore()?.getItem(STORAGE_KEY));
}

export function clearStashedPairingToken(): void {
  sessionStore()?.removeItem(STORAGE_KEY);
}
