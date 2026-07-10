import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import * as session from "../auth/session";
import { ProjectDashboardPage } from "./ProjectDashboardPage";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      me: vi.fn(),
      getProject: vi.fn(),
      listDocuments: vi.fn(),
      createDocument: vi.fn(),
      deleteDocument: vi.fn(),
      updateDocument: vi.fn(),
      deleteProject: vi.fn(),
      updateProject: vi.fn(),
    },
  };
});

function renderProjectDashboard() {
  window.history.replaceState({}, "", "/projects/project-1");
  return render(<ProjectDashboardPage />);
}

describe("ProjectDashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
    vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
    vi.mocked(api.me).mockResolvedValue({
      id: "user-1",
      email: "dev@example.com",
      username: "dev",
      created_at: "2026-01-01T00:00:00Z",
    });
    vi.mocked(api.getProject).mockResolvedValue({
      id: "project-1",
      name: "Test Project",
      slug: "test-project",
      guidelines: null,
      owner_id: "user-1",
      document_count: 1,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    });
    vi.mocked(api.listDocuments).mockResolvedValue([
      {
        id: "doc-1",
        project_id: "project-1",
        name: "Grec 1360",
        workflow: "draft",
        part_count: 3,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);
  });

  it("shows an unavailable state instead of document actions when project access is rejected", async () => {
    vi.mocked(api.getProject).mockRejectedValue(new ApiError("Forbidden", 403));
    vi.mocked(api.listDocuments).mockResolvedValue([]);

    renderProjectDashboard();

    expect(await screen.findByText("Project unavailable")).toBeTruthy();
    expect(
      screen.getByText("This project is not available to your account."),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: /new document/i })).toBeNull();
  });

  it("lets a project member delete a document from the table", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(api.deleteDocument).mockResolvedValue(undefined);

    renderProjectDashboard();

    await screen.findByRole("heading", { name: "Test Project" });
    fireEvent.click(
      screen.getByRole("button", { name: /delete document grec 1360/i }),
    );

    await waitFor(() => {
      expect(api.deleteDocument).toHaveBeenCalledWith("project-1", "doc-1");
    });
  });

  it("lets the owner edit the project from the dashboard header", async () => {
    vi.mocked(api.updateProject).mockResolvedValue({
      id: "project-1",
      name: "ByzantineGreekCorpus",
      slug: "byzantine-greek-corpus",
      guidelines: "Updated notes",
      owner_id: "user-1",
      document_count: 1,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-02T00:00:00Z",
    });

    renderProjectDashboard();

    fireEvent.click(
      await screen.findByRole("button", {
        name: /test project, click to edit/i,
      }),
    );
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "ByzantineGreekCorpus" },
    });
    fireEvent.change(screen.getByLabelText("Guidelines"), {
      target: { value: "Updated notes" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(api.updateProject).toHaveBeenCalledWith("project-1", {
        name: "ByzantineGreekCorpus",
        slug: "byzantinegreekcorpus",
        guidelines: "Updated notes",
      });
    });
    expect(
      screen.getByRole("heading", { name: "ByzantineGreekCorpus" }),
    ).toBeTruthy();
  });

  it("redirects to login when the session is unauthorized", async () => {
    vi.mocked(api.getProject).mockRejectedValue(
      new ApiError("Unauthorized", 401),
    );

    renderProjectDashboard();

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(screen.queryByText("Project unavailable")).toBeNull();
  });

  it("redirects to login when no access token is present", async () => {
    vi.spyOn(session, "hasAccessToken").mockReturnValue(false);

    renderProjectDashboard();

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(api.getProject).not.toHaveBeenCalled();
  });
});
