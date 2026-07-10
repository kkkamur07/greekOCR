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

describe("LoginPage", () => {
  beforeEach(() => {
    clearAccessToken();
    vi.clearAllMocks();
  });

  it("signs in and returns the user to the protected page they requested", async () => {
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
    await waitFor(() => {
      expect(getAccessToken()).toBe("jwt-token");
    });
  });
});
