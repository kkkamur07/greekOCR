import { invalidateAuthGetCache } from "../api/getCache";
import { clearImageCache } from "../api/imageCache";
import { queryClient } from "../api/queryClient";

let accessToken: string | null = null;

/**
 * The browser session's CSRF token, as the server handed it back in the body.
 *
 * A second copy of what the readable `greekocr-csrf` cookie carries, kept for
 * the case where script cannot read that cookie. The cookie lives on
 * `.nomicous.com` so that `app.nomicous.com` can read a value the API set on
 * `api.nomicous.com`; a browser that declines that sibling-subdomain read
 * leaves the client with no way to build `X-CSRF-Token`, and every
 * `POST /auth/refresh` answers 403. Held in memory rather than in
 * `localStorage`, for the same reason the access token is: a value that
 * survives the tab outlives the session it belongs to.
 */
let csrfToken: string | null = null;

export function getAccessToken(): string | null {
  return accessToken;
}

export function getCsrfToken(): string | null {
  return csrfToken;
}

export function setCsrfToken(token: string): void {
  csrfToken = token;
}

export function clearCsrfToken(): void {
  csrfToken = null;
}

export function setAccessToken(token: string): void {
  accessToken = token;
  // Retained reads belong to whoever was signed in when they were made, so a new
  // session must not be able to see them.
  invalidateAuthGetCache();
  queryClient.clear();
}

export function clearAccessToken(): void {
  accessToken = null;
  // Both credentials belong to the one session that just ended. Leaving the
  // CSRF token behind would let the next sign-in start by presenting the
  // previous session's token.
  csrfToken = null;
  invalidateAuthGetCache();
  queryClient.clear();
  clearImageCache();
}
