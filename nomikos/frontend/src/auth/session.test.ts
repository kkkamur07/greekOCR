import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/errors";
import {
  clearLoginRedirectGuard,
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
  redirectToLogin,
} from "./session";
import { clearAccessToken, setAccessToken } from "./storage";

// The two callers navigate by different mechanisms: `navigateToLogin` uses the Next
// router, `redirectToLogin` uses `window.location.assign`. A test that watches only the
// router therefore cannot see the second one at all -- which is how the guard this file
// exists to pin was, for a while, asserted by nothing.
//
// jsdom's `location` is unforgeable, so `vi.spyOn(window.location, "assign")` throws
// "Cannot redefine property". Replacing the whole object does work, and has the useful
// side effect of letting a test choose the pathname/search/hash the callback URL is
// built from.
const realLocation = Object.getOwnPropertyDescriptor(window, "location");

function stubLocation(href: string): ReturnType<typeof vi.fn> {
  const parsed = new URL(href, "http://localhost");
  const assign = vi.fn();
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      pathname: parsed.pathname,
      search: parsed.search,
      hash: parsed.hash,
      assign,
    },
  });
  return assign;
}

describe("login redirects", () => {
  beforeEach(() => {
    clearLoginRedirectGuard();
    clearAccessToken();
  });

  afterEach(() => {
    clearLoginRedirectGuard();
    if (realLocation) Object.defineProperty(window, "location", realLocation);
  });

  it("lets the React caller through and suppresses the API caller behind it", () => {
    const assign = stubLocation("/projects/project-1");
    const router = { replace: vi.fn() };

    navigateToLogin(router);
    redirectToLogin();

    expect(router.replace).toHaveBeenCalledExactlyOnceWith(
      "/login?callbackUrl=%2Fprojects%2Fproject-1",
    );
    // The assertion the old test could not make: the second caller really was stopped.
    expect(assign).not.toHaveBeenCalled();
  });

  it("lets the API caller through and suppresses the React caller behind it", () => {
    const assign = stubLocation("/projects/project-1");
    const router = { replace: vi.fn() };

    redirectToLogin();
    navigateToLogin(router);

    expect(assign).toHaveBeenCalledExactlyOnceWith(
      "/login?callbackUrl=%2Fprojects%2Fproject-1",
    );
    expect(router.replace).not.toHaveBeenCalled();
  });

  it("carries the query and fragment into the callback URL", () => {
    stubLocation("/documents/7?page=3#line-12");
    const router = { replace: vi.fn() };

    navigateToLogin(router);

    // Dropping search or hash silently loses the reader's place on the way back.
    expect(router.replace).toHaveBeenCalledWith(
      "/login?callbackUrl=%2Fdocuments%2F7%3Fpage%3D3%23line-12",
    );
  });

  it.each(["/login", "/register"])(
    "never redirects away from %s",
    (pathname) => {
      const assign = stubLocation(pathname);
      const router = { replace: vi.fn() };

      navigateToLogin(router);
      redirectToLogin();

      // Without this suppression a 401 raised by the login form itself sends the page
      // back to the login form, forever.
      expect(router.replace).not.toHaveBeenCalled();
      expect(assign).not.toHaveBeenCalled();
    },
  );

  it("clears the token even when the redirect itself is suppressed", () => {
    stubLocation("/login");
    setAccessToken("stale-token");

    redirectToLogin();

    // The clear happens before the guard check: a token that failed must not survive
    // just because we were already on the login page.
    expect(hasAccessToken()).toBe(false);
  });
});

describe("session predicates", () => {
  beforeEach(() => {
    clearAccessToken();
  });

  it("reports a token only when one is actually held", () => {
    expect(hasAccessToken()).toBe(false);
    setAccessToken("a-token");
    expect(hasAccessToken()).toBe(true);
    clearAccessToken();
    expect(hasAccessToken()).toBe(false);
  });

  it("treats a whitespace-only token as no token", () => {
    setAccessToken("   ");
    expect(hasAccessToken()).toBe(false);
  });

  it("recognises only a 401 ApiError as unauthorized", () => {
    expect(isUnauthorized(new ApiError("nope", 401))).toBe(true);
    expect(isUnauthorized(new ApiError("nope", 403))).toBe(false);
    expect(isUnauthorized(new Error("401"))).toBe(false);
    expect(isUnauthorized(null)).toBe(false);
  });
});
