import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ProjectResponse } from "../api/client";
import { queryClient, taggedMeta } from "../api/queryClient";
import { ApiError } from "../api/errors";
import { resourceTags } from "../api/resources";
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

const PROJECT: ProjectResponse = {
  id: "project-1",
  name: "Test Project",
  slug: "test-project",
  guidelines: null,
  owner_id: "user-1",
  document_count: 1,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

/**
 * The project, stored the way the platform stores it: a PATCH changes it and
 * every later GET reflects the change.
 *
 * A canned `getProject` would hide what this page has to get right. The panel
 * patches its own view and then declares the write, which refetches; if the
 * refetch answered with the pre-write project forever, a mutation that
 * invalidated nothing would look identical to one that did.
 */
function seedProject() {
  let stored: ProjectResponse = PROJECT;
  vi.mocked(api.getProject).mockImplementation(async () => stored);
  vi.mocked(api.updateProject).mockImplementation(
    async (
      _projectId: string,
      patch: { name?: string; slug?: string; guidelines?: string | null },
    ) => {
      stored = { ...stored, ...patch };
      return stored;
    },
  );
}

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
    seedProject();
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

  it("declares every cached read of the project stale after a rename", async () => {
    // `includeArchived` is part of this page's query key, so the view the
    // researcher is not looking at is a second cache entry with its own copy of
    // the project. The panel patches only the entry on screen; nothing else
    // here reads this one, which is why a rename used to come back undone the
    // moment Show archived was ticked.
    const archivedKey = ["project-dashboard", "project-1", true];
    await queryClient.fetchQuery({
      queryKey: archivedKey,
      queryFn: () => Promise.resolve({ project: PROJECT }),
      meta: taggedMeta([resourceTags.project("project-1")]),
    });

    renderProjectDashboard();
    await screen.findByRole("heading", { name: "Test Project" });

    fireEvent.click(
      screen.getByRole("button", { name: /test project, click to edit/i }),
    );
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "ByzantineGreekCorpus" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));
    await screen.findByRole("heading", { name: "ByzantineGreekCorpus" });

    await waitFor(() => {
      expect(queryClient.getQueryState(archivedKey)?.isInvalidated).toBe(true);
    });
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
});
