import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { testRouter } from "../../vitest.setup";

import { AuthProvider } from "../auth/AuthProvider";
import { clearLoginRedirectGuard } from "../auth/session";
import { clearAccessToken, getAccessToken } from "../auth/storage";
import { toast } from "../components/ui/toast";
import { LoginPage } from "./LoginPage";

// `useAuthSession` is not stubbed here. The page's whole job is to turn a successful
// sign-in into a session, and a mocked `establish` spy proves only that the page called
// something -- not that a token was stored. The real provider is rendered and the
// network is stubbed at `fetch`, so `getAccessToken()` is the assertion.
vi.mock("../components/ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

type Route = (init: RequestInit | undefined) => Promise<Response> | Response;

function stubApi(routes: Record<string, Route>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    for (const [suffix, route] of Object.entries(routes)) {
      if (url.endsWith(suffix)) return route(init);
    }
    throw new Error(`unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** A provider that has already settled into `anonymous`. */
const anonymous: Record<string, Route> = {
  "/auth/refresh": () => new Response(null, { status: 401 }),
};

function renderLogin() {
  return render(
    <AuthProvider>
      <LoginPage />
    </AuthProvider>,
  );
}

describe("LoginPage", () => {
  beforeEach(() => {
    clearAccessToken();
    clearLoginRedirectGuard();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not show the sign-in form while session restore is in flight", async () => {
    let settle!: (response: Response) => void;
    stubApi({
      "/auth/refresh": () =>
        new Promise<Response>((resolve) => {
          settle = resolve;
        }),
    });

    renderLogin();

    expect(screen.queryByRole("heading", { name: /sign in/i })).toBeNull();
    expect(screen.getByText(/restoring your session/i)).toBeTruthy();
    expect(testRouter().replace).not.toHaveBeenCalled();

    // Settle before leaving: `refreshAccessToken` memoises the in-flight call on a
    // module singleton, so a pending promise would still be the "current" refresh for
    // the next test in this file.
    settle(new Response(null, { status: 401 }));
    await screen.findByRole("heading", { name: /sign in/i });
  });

  it("redirects an already-authenticated reader away without flashing the form", async () => {
    stubApi({
      "/auth/refresh": () =>
        Response.json({ access_token: "restored", token_type: "bearer" }),
    });
    window.history.replaceState({}, "", "/login?callbackUrl=%2Fprojects%2Fproject-1");

    renderLogin();

    expect(screen.queryByRole("heading", { name: /sign in/i })).toBeNull();
    await waitFor(() => {
      expect(testRouter().replace).toHaveBeenCalledWith("/projects/project-1");
    });
  });

  it("signs in, stores the token, and returns the reader to the page they asked for", async () => {
    stubApi({
      ...anonymous,
      "/auth/login": () =>
        Response.json({ access_token: "jwt-token", token_type: "bearer" }),
    });
    window.history.replaceState(
      {},
      "",
      "/login?callbackUrl=%2Fprojects%2Fproject-1%2Fdocuments%2Fdoc-1%2Fparts%2Fpart-1%3Fpanel%3Dhistory",
    );
    renderLogin();
    await screen.findByRole("heading", { name: /sign in/i });

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "researcher@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct-password" },
    });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(testRouter().replace).toHaveBeenCalledWith(
        "/projects/project-1/documents/doc-1/parts/part-1?panel=history",
      );
    });
    // The real outcome of a sign-in: every later request is signed with this.
    expect(getAccessToken()).toBe("jwt-token");
  });

  it("tells the reader when the credentials are rejected, and stores nothing", async () => {
    stubApi({
      ...anonymous,
      "/auth/login": () =>
        Response.json(
          { error: { message: "Incorrect email or password" } },
          { status: 401 },
        ),
    });
    renderLogin();
    await screen.findByRole("heading", { name: /sign in/i });

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "researcher@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "wrong-password" },
    });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    // Without this, a wrong password produces no visible feedback at all and the form
    // simply sits there.
    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect(vi.mocked(toast.error).mock.calls[0]?.[0]).toMatch(/incorrect|failed/i);
    expect(getAccessToken()).toBeNull();
    expect(testRouter().replace).not.toHaveBeenCalled();
    // The button must come back so the reader can try again.
    expect(
      screen.getByRole("button", { name: /sign in/i }).hasAttribute("disabled"),
    ).toBe(false);
  });
});
