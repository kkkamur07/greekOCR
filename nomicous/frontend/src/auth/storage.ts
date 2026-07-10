import { invalidateAuthGetCache } from '../api/getCache';
import { clearImageCache } from '../api/imageCache';

let accessToken: string | null = null;

export function getAccessToken(): string | null {
  return accessToken;
}

export function setAccessToken(token: string): void {
  accessToken = token;
  invalidateAuthGetCache();
}

export function clearAccessToken(): void {
  accessToken = null;
  invalidateAuthGetCache();
  clearImageCache();
}
