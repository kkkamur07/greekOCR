import { ApiError } from "../api/errors";
import { clearAccessToken, getAccessToken } from "./storage";

let loginRedirectInFlight = false;

export function hasAccessToken(): boolean {
  const token = getAccessToken();
  return typeof token === "string" && token.trim().length > 0;
}

export function clearLoginRedirectGuard(): void {
  loginRedirectInFlight = false;
}

function beginLoginRedirect(): boolean {
  if (
    loginRedirectInFlight ||
    typeof window === "undefined" ||
    window.location.pathname === "/login" ||
    window.location.pathname === "/register"
  ) {
    return false;
  }
  loginRedirectInFlight = true;
  return true;
}

/** Full-page redirect for API-layer auth failures outside React. */
export function redirectToLogin(): void {
  clearAccessToken();
  if (!beginLoginRedirect()) return;
  const callbackUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  window.location.assign(
    `/login?callbackUrl=${encodeURIComponent(callbackUrl)}`,
  );
}

export function navigateToLogin(
  router: Pick<{ replace: (href: string) => void }, "replace">,
): void {
  clearAccessToken();
  if (!beginLoginRedirect()) return;
  const callbackUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  router.replace(`/login?callbackUrl=${encodeURIComponent(callbackUrl)}`);
}

export function isUnauthorized(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401;
}
