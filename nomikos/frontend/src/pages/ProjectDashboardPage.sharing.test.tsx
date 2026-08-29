import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ProjectResponse } from "../api/client";
import * as session from "../auth/session";
import { ProjectDashboardPage } from "./ProjectDashboardPage";

/**
 * The Share control lives in its own file rather than in
 * ProjectDashboardPage.test.tsx: that file shares module-level request state
 * across its tests, and a test appended after the unauthorized-redirect case
 * never resolves its first read.
 */
vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      me: vi.fn(),
      getProject: vi.fn(),
      listDocuments: vi.fn(),
      listProjectCollaborators: vi.fn(),
    },
  };
});

const PROJECT: ProjectResponse = {
  id: "project-1",
  name: "Test Project",
  slug: "test-project",
  guidelines: null,
  owner_id: "user-1",
  document_count: 0,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

describe("ProjectDashboardPage sharing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // useParams is derived from the URL by vitest.setup.ts, which resets it to
    // "/" before each test. Without this the page has no projectId, reads
    // nothing, and sits on its loading state forever.
    window.history.replaceState({}, "", "/projects/project-1");
    vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
    vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
    vi.mocked(api.getProject).mockResolvedValue(PROJECT);
    vi.mocked(api.listDocuments).mockResolvedValue([]);
    vi.mocked(api.listProjectCollaborators).mockResolvedValue([]);
    vi.mocked(api.me).mockResolvedValue({
      id: "user-1",
      email: "dev@example.com",
      username: "dev",
      created_at: "2026-01-01T00:00:00Z",
    });
  });

  it("offers a visible Share button that reveals the sharing controls", async () => {
    render(<ProjectDashboardPage />);
    await screen.findByRole("heading", { name: "Test Project" });

    // Sharing is not on the page until asked for. Before this button existed
    // the only way in was clicking the project title, which renders as a
    // heading and advertises nothing.
    expect(screen.queryByLabelText("Email or username")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Share" }));

    const box = await screen.findByLabelText("Email or username");
    expect(box).toBeInTheDocument();
    // Focused, so "Share" lands you ready to type.
    expect(box).toHaveFocus();
  });

  it("keeps the panel open when Share is clicked a second time", async () => {
    render(<ProjectDashboardPage />);
    await screen.findByRole("heading", { name: "Test Project" });

    const share = screen.getByRole("button", { name: "Share" });
    fireEvent.click(share);
    await screen.findByLabelText("Email or username");
    screen.getByLabelText("Email or username").blur();

    // The panel closes on any click outside itself and this button is outside
    // it, so a naive toggle would flicker it shut on the second press.
    fireEvent.click(share);

    expect(screen.getByLabelText("Email or username")).toBeInTheDocument();
    expect(screen.getByLabelText("Email or username")).toHaveFocus();
  });

  it("does not offer Share to someone who is not the owner", async () => {
    vi.mocked(api.me).mockResolvedValue({
      id: "someone-else",
      email: "collab@example.com",
      username: "collab",
      created_at: "2026-01-01T00:00:00Z",
    });

    render(<ProjectDashboardPage />);
    await screen.findByRole("heading", { name: "Test Project" });

    expect(screen.queryByRole("button", { name: "Share" })).toBeNull();
  });
});
