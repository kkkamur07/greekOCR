import { invalidateAuthGetCache } from "../api/getCache";
import { clearImageCache } from "../api/imageCache";
import { clearResourceCache } from "../api/resourceCache";

let accessToken: string | null = null;

export function getAccessToken(): string | null {
  return accessToken;
}

export function setAccessToken(token: string): void {
  accessToken = token;
  // Retained reads belong to whoever was signed in when they were made, so a new
  // session must not be able to see them.
  invalidateAuthGetCache();
  clearResourceCache();
}

export function clearAccessToken(): void {
  accessToken = null;
  invalidateAuthGetCache();
  clearResourceCache();
  clearImageCache();
}
