import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiRequest } from "../api/client";
import { AuthProvider, useAuthSession } from "./AuthProvider";
import { clearAccessToken, getAccessToken } from "./storage";

function SessionStatus() {
  const { status } = useAuthSession();
  return <output>{status}</output>;
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("AuthProvider refresh recovery", () => {
  beforeEach(() => {
    clearAccessToken();
  });

  it("shares bootstrap refresh with a protected request", async () => {
    let resolveRefresh: ((response: Response) => void) | undefined;
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/auth/refresh")) {
        return new Promise<Response>((resolve) => {
          resolveRefresh = resolve;
        });
      }
      const token = new Headers(init?.headers).get("Authorization");
      return Promise.resolve(
        token === "Bearer restored-token"
          ? jsonResponse({ id: "project-1" })
          : new Response(null, { status: 401 }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <AuthProvider>
        <SessionStatus />
      </AuthProvider>,
    );
    const request = apiRequest<{ id: string }>("/projects/project-1");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    resolveRefresh?.(jsonResponse({ access_token: "restored-token" }));

    await expect(request).resolves.toEqual({ id: "project-1" });
    await waitFor(() =>
      expect(screen.getByText("authenticated")).toBeInTheDocument(),
    );
    expect(
      fetchMock.mock.calls.filter(([url]) => url.endsWith("/auth/refresh")),
    ).toHaveLength(1);
    expect(getAccessToken()).toBe("restored-token");
  });
});
