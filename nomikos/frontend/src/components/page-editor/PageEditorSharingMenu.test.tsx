import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import { PageEditorSharingMenu } from "./PageEditorSharingMenu";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      updateDocument: vi.fn(),
    },
  };
});

const mockedUpdateDocument = api.updateDocument as ReturnType<typeof vi.fn>;

function renderMenu(
  workflow: "draft" | "published" | "archived" = "draft",
  publicShareToken: string | null = null,
) {
  const onWorkflowChange = vi.fn();
  render(
    <div role="menu">
      <PageEditorSharingMenu
        projectId="project-1"
        documentId="doc-1"
        workflow={workflow}
        publicShareToken={publicShareToken}
        onWorkflowChange={onWorkflowChange}
      />
    </div>,
  );
  return { onWorkflowChange };
}

describe("PageEditorSharingMenu", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedUpdateDocument.mockResolvedValue({
      id: "doc-1",
      project_id: "project-1",
      name: "Grec 1360",
      workflow: "published",
      created_at: "2026-06-16T10:00:00Z",
      updated_at: "2026-06-16T10:00:00Z",
      part_count: 1,
    });
  });

  it("publishes a draft document from the sharing section", async () => {
    const { onWorkflowChange } = renderMenu("draft");

    fireEvent.click(
      screen.getByRole("menuitem", { name: /^publish document$/i }),
    );

    await waitFor(() => {
      expect(mockedUpdateDocument).toHaveBeenCalledWith("project-1", "doc-1", {
        workflow: "published",
      });
      expect(onWorkflowChange).toHaveBeenCalledWith("published");
    });
  });

  it("shows the public link, with its token, when the document is published", () => {
    renderMenu("published", "share-token-1");

    expect(screen.getByLabelText(/public document url/i)).toHaveValue(
      `${window.location.origin}/public/projects/project-1/documents/doc-1?t=share-token-1`,
    );
    expect(
      screen.getByRole("link", { name: /open public view/i }),
    ).toHaveAttribute(
      "href",
      "/public/projects/project-1/documents/doc-1?t=share-token-1",
    );
  });

  it("tells a collaborator the owner has to hand out the link, rather than offer one that 404s", () => {
    renderMenu("published", null);

    expect(
      screen.getByText(/only the project owner can get the public share link/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText(/public document url/i),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: /open public view/i }),
    ).not.toBeInTheDocument();
  });
});
