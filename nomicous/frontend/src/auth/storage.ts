import { invalidateAuthGetCache } from "../api/getCache";
import { clearImageCache } from "../api/imageCache";
import { queryClient } from "../api/queryClient";

let accessToken: string | null = null;

export function getAccessToken(): string | null {
  return accessToken;
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
  invalidateAuthGetCache();
  queryClient.clear();
  clearImageCache();
}
