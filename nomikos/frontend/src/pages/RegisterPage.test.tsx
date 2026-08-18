import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testRouter } from "../../vitest.setup";

import { api } from "../api/client";
import { clearAccessToken, getAccessToken } from "../auth/storage";
import { RegisterPage } from "./RegisterPage";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      register: vi.fn(),
    },
  };
});

describe("RegisterPage", () => {
  beforeEach(() => {
    clearAccessToken();
    vi.clearAllMocks();
  });

  it("creates an account and returns the user to the protected page they requested", async () => {
    vi.mocked(api.register).mockResolvedValue({
      access_token: "new-user-token",
      token_type: "bearer",
    });

    window.history.replaceState(
      {},
      "",
      "/register?callbackUrl=%2Fprojects%2Fproject-1%2Fdocuments%2Fdoc-1%2Fparts%2Fpart-1%3Fpanel%3Dhistory",
    );
    render(<RegisterPage />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "new.researcher@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "new-researcher" },
    });
    fireEvent.change(screen.getByLabelText(/^password/i), {
      target: { value: "correct-password" },
    });
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => {
      expect(testRouter().replace).toHaveBeenCalledWith(
        "/projects/project-1/documents/doc-1/parts/part-1?panel=history",
      );
    });
    await waitFor(() => {
      expect(getAccessToken()).toBe("new-user-token");
    });
  });
});
