import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testRouter } from "../../vitest.setup";

import { api } from "../api/client";
import { clearAccessToken, getAccessToken } from "../auth/storage";
import { LoginPage } from "./LoginPage";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      login: vi.fn(),
    },
  };
});

vi.mock("../auth/AuthProvider", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../auth/AuthProvider")>();
  return {
    ...actual,
    useAuthSession: vi.fn(),
  };
});

import { useAuthSession } from "../auth/AuthProvider";

describe("LoginPage", () => {
  beforeEach(() => {
    clearAccessToken();
    vi.clearAllMocks();
    vi.mocked(useAuthSession).mockReturnValue({
      status: "anonymous",
      establish: vi.fn(),
      logout: vi.fn(),
    });
  });

  it("does not show the sign-in form while session restore is in flight", () => {
    vi.mocked(useAuthSession).mockReturnValue({
      status: "restoring",
      establish: vi.fn(),
      logout: vi.fn(),
    });

    render(<LoginPage />);

    expect(screen.queryByRole("heading", { name: /sign in/i })).toBeNull();
    expect(screen.getByText(/restoring your session/i)).toBeTruthy();
    expect(testRouter().replace).not.toHaveBeenCalled();
  });

  it("redirects authenticated users away from login without flashing the form", async () => {
    vi.mocked(useAuthSession).mockReturnValue({
      status: "authenticated",
      establish: vi.fn(),
      logout: vi.fn(),
    });
    window.history.replaceState(
      {},
      "",
      "/login?callbackUrl=%2Fprojects%2Fproject-1",
    );

    render(<LoginPage />);

    expect(screen.queryByRole("heading", { name: /sign in/i })).toBeNull();
    await waitFor(() => {
      expect(testRouter().replace).toHaveBeenCalledWith("/projects/project-1");
    });
  });

  it("signs in and returns the user to the protected page they requested", async () => {
    const establish = vi.fn();
    vi.mocked(useAuthSession).mockReturnValue({
      status: "anonymous",
      establish,
      logout: vi.fn(),
    });
    vi.mocked(api.login).mockResolvedValue({
      access_token: "jwt-token",
      token_type: "bearer",
    });

    window.history.replaceState(
      {},
      "",
      "/login?callbackUrl=%2Fprojects%2Fproject-1%2Fdocuments%2Fdoc-1%2Fparts%2Fpart-1%3Fpanel%3Dhistory",
    );
    render(<LoginPage />);

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
    expect(establish).toHaveBeenCalledWith("jwt-token");
  });
});
