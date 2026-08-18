import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { testRouter } from "../../vitest.setup";
import { AuthProvider } from "../auth/AuthProvider";
import { clearLoginRedirectGuard } from "../auth/session";
import { clearAccessToken, setAccessToken } from "../auth/storage";
import { ProtectedRoute } from "./ProtectedRoute";

// Nothing here is mocked but `fetch`. Stubbing `useAuthSession` used to mean the real
// provider never ran, and stubbing `navigateToLogin` meant the assertion could not see
// where the user was actually sent -- passing a dead router to it left the suite green.
// Status is driven the way production drives it: the bootstrap refresh call.
function stubRefresh(response: () => Promise<Response>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/auth/refresh")) return response();
    throw new Error(`unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderProtected() {
  return render(
    <AuthProvider>
      <ProtectedRoute>
        <div>Projects content</div>
      </ProtectedRoute>
    </AuthProvider>,
  );
}

describe("ProtectedRoute", () => {
  beforeEach(() => {
    // Both are module-level singletons that outlive a test.
    clearAccessToken();
    clearLoginRedirectGuard();
    window.history.replaceState({}, "", "/projects");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends an anonymous reader to sign in, carrying where they were", async () => {
    stubRefresh(async () => new Response(null, { status: 401 }));

    renderProtected();

    await waitFor(() => {
      expect(testRouter().replace).toHaveBeenCalledExactlyOnceWith(
        "/login?callbackUrl=%2Fprojects",
      );
    });
    expect(screen.queryByText("Projects content")).toBeNull();
  });

  it("shows restoring chrome and never navigates while the refresh is in flight", async () => {
    // A refresh that has not settled is exactly the window in which a premature redirect
    // would throw away a session that was about to be restored.
    let settle!: (response: Response) => void;
    stubRefresh(
      () =>
        new Promise<Response>((resolve) => {
          settle = resolve;
        }),
    );

    renderProtected();

    expect(await screen.findByText(/restoring your session/i)).toBeTruthy();
    expect(screen.queryByText("Projects content")).toBeNull();
    expect(testRouter().replace).not.toHaveBeenCalled();

    // Settle before leaving. `refreshAccessToken` memoises the in-flight call on a
    // module-level `refreshPromise` that is only cleared in its `.finally`, so a
    // promise left pending here is still the "current" refresh for every later test in
    // the file -- which is a leak, and also the reason the redirect below is worth
    // asserting: restoring must resolve to a decision, not stall.
    settle(new Response(null, { status: 401 }));
    await waitFor(() => {
      expect(testRouter().replace).toHaveBeenCalledWith(
        "/login?callbackUrl=%2Fprojects",
      );
    });
  });

  it("renders children once the refresh establishes a session", async () => {
    stubRefresh(async () =>
      Response.json({ access_token: "refreshed-token" }, { status: 200 }),
    );

    renderProtected();

    expect(await screen.findByText("Projects content")).toBeTruthy();
    expect(testRouter().replace).not.toHaveBeenCalled();
  });

  it("renders children immediately for a reader who already holds a token", () => {
    setAccessToken("existing-token");
    const fetchMock = stubRefresh(
      async () => new Response(null, { status: 401 }),
    );

    renderProtected();

    expect(screen.getByText("Projects content")).toBeTruthy();
    expect(testRouter().replace).not.toHaveBeenCalled();
    // A held token must not cost a refresh round trip on every mount.
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
