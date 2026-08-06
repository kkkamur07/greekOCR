/**
 * Where `X-CSRF-Token` comes from.
 *
 * The server sends the session's CSRF token twice - in the auth response body
 * and in the `greekocr-csrf` cookie - because the cookie is the fragile copy:
 * it is set by `api.nomicous.com` for `.nomicous.com` purely so that script on
 * `app.nomicous.com` can read it back, and a browser that declines that
 * sibling-subdomain read leaves the client unable to build the header at all.
 * These tests pin both channels and the order they are consulted in.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, apiRequest, refreshAccessToken } from "./client";
import { ApiError } from "./errors";
import { clearLoginRedirectGuard } from "../auth/session";
import { clearAccessToken, getCsrfToken } from "../auth/storage";

const CSRF_COOKIE = "greekocr-csrf";

function setCsrfCookie(value: string): void {
  document.cookie = `${CSRF_COOKIE}=${encodeURIComponent(value)}`;
}

function clearCsrfCookie(): void {
  document.cookie = `${CSRF_COOKIE}=; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** The CSRF header each recorded call carried, in order. */
function sentTokens(fetchMock: {
  mock: { calls: unknown[][] };
}): Array<string | null> {
  return fetchMock.mock.calls.map(([, init]) =>
    new Headers((init as RequestInit | undefined)?.headers).get("X-CSRF-Token"),
  );
}

describe("CSRF token delivery", () => {
  beforeEach(() => {
    clearAccessToken();
    clearCsrfCookie();
    clearLoginRedirectGuard();
  });

  afterEach(() => {
    clearCsrfCookie();
    vi.unstubAllGlobals();
  });

  it("reads the cookie when no auth response has been seen in this tab", async () => {
    // A session established before this code shipped, or a page that has not
    // signed in or refreshed yet: the cookie is the only channel there is.
    setCsrfCookie("cookie-token");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await apiRequest("/projects", { method: "POST", body: {} });

    expect(sentTokens(fetchMock)).toEqual(["cookie-token"]);
  });

  it("prefers the token the server returned in the body over the cookie", async () => {
    // The whole point: a browser that will not let script read the cookie still
    // has this copy, so the header can be built without touching `document.cookie`.
    setCsrfCookie("cookie-token");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt", csrf_token: "body-token" }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await api.login({ email: "a@b.com", password: "secret-password" });
    await apiRequest("/projects", { method: "POST", body: {} });

    expect(getCsrfToken()).toBe("body-token");
    expect(sentTokens(fetchMock)[1]).toBe("body-token");
  });

  it("builds the header with no readable cookie at all", async () => {
    // Safari with the sibling-subdomain read blocked, modelled as the cookie
    // simply not being visible to script. Before the body copy existed this
    // request went out with no header and was answered 403.
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt", csrf_token: "body-token" }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await api.register({
      email: "a@b.com",
      username: "a",
      password: "secret-password",
    });
    await apiRequest("/projects", { method: "POST", body: {} });

    expect(document.cookie).not.toContain(CSRF_COOKIE);
    expect(sentTokens(fetchMock)[1]).toBe("body-token");
  });

  it("tolerates an API that predates the body field", async () => {
    // Frontend and API deploy separately, so a new client can meet an old server.
    setCsrfCookie("cookie-token");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ access_token: "jwt" }))
      .mockResolvedValueOnce(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await api.login({ email: "a@b.com", password: "secret-password" });
    await apiRequest("/projects", { method: "POST", body: {} });

    expect(getCsrfToken()).toBeNull();
    expect(sentTokens(fetchMock)[1]).toBe("cookie-token");
  });

  it("keeps the rotated token after a refresh", async () => {
    setCsrfCookie("first-token");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt-2", csrf_token: "second-token" }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await refreshAccessToken();
    await apiRequest("/projects", { method: "POST", body: {} });

    expect(sentTokens(fetchMock)).toEqual(["first-token", "second-token"]);
  });

  it("drops the token when the session ends", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt", csrf_token: "body-token" }),
      );
    vi.stubGlobal("fetch", fetchMock);
    await api.login({ email: "a@b.com", password: "secret-password" });

    clearAccessToken();

    // Otherwise the next sign-in would open by presenting the dead session's token.
    expect(getCsrfToken()).toBeNull();
  });
});

describe("stale in-memory CSRF token", () => {
  beforeEach(() => {
    clearAccessToken();
    clearCsrfCookie();
    clearLoginRedirectGuard();
  });

  afterEach(() => {
    clearCsrfCookie();
    vi.unstubAllGlobals();
  });

  /** Sign in, then let another tab rotate the shared cookie out from under us. */
  async function signInThenRotateElsewhere(): Promise<void> {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt", csrf_token: "mine" }),
      );
    vi.stubGlobal("fetch", fetchMock);
    await api.login({ email: "a@b.com", password: "secret-password" });
    setCsrfCookie("rotated-by-another-tab");
  }

  it("retries a refused refresh from the cookie", async () => {
    // The in-memory token is this tab's; `/auth/refresh` rotates the session's
    // token for every tab at once. Without this retry, a second open tab would
    // be answered 403 and bounce the reader to the sign-in page - a regression
    // for every browser that reads the cookie fine today.
    await signInThenRotateElsewhere();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 403 }))
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt-2", csrf_token: "fresh" }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(refreshAccessToken()).resolves.toMatchObject({
      access_token: "jwt-2",
    });

    expect(sentTokens(fetchMock)).toEqual(["mine", "rotated-by-another-tab"]);
    expect(getCsrfToken()).toBe("fresh");
  });

  it("retries a refused logout the same way", async () => {
    // A tab that cannot sign out for real leaves the session live on the server.
    await signInThenRotateElsewhere();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 403 }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.logout()).resolves.toBeUndefined();

    expect(sentTokens(fetchMock)).toEqual(["mine", "rotated-by-another-tab"]);
  });

  it("does not retry when the cookie cannot tell it anything new", async () => {
    // No readable cookie (the Safari case) or the same value: a second request
    // would fail identically, so the 403 is surfaced straight away.
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: "jwt", csrf_token: "mine" }),
      )
      .mockResolvedValue(new Response(null, { status: 403 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.login({ email: "a@b.com", password: "secret-password" });

    await expect(refreshAccessToken()).rejects.toMatchObject({ status: 403 });
    setCsrfCookie("mine");
    await expect(refreshAccessToken()).rejects.toBeInstanceOf(ApiError);

    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("does not retry a failure that is not a CSRF refusal", async () => {
    await signInThenRotateElsewhere();
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(refreshAccessToken()).rejects.toMatchObject({ status: 401 });

    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
