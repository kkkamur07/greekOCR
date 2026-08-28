import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testRouter } from "../../vitest.setup";

import {
  api,
  type DocumentResponse,
  type DocumentWithPartsResponse,
} from "../api/client";
import { ApiError } from "../api/errors";
import { queryClient, taggedMeta } from "../api/queryClient";
import { resourceTags } from "../api/resources";
import * as session from "../auth/session";
import { DocumentDetailPage } from "./DocumentDetailPage";

vi.mock("../components/AuthenticatedImage", () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock("../components/document/JobsNotice", () => ({
  JobsNotice: () => <div>Jobs panel</div>,
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getDocument: vi.fn(),
      me: vi.fn(),
      getProject: vi.fn(),
      uploadPart: vi.fn(),
      reorderParts: vi.fn(),
      deletePart: vi.fn(),
      updatePartReviewStatus: vi.fn(),
      updatePartsPublished: vi.fn(),
      updateDocument: vi.fn(),
    },
  };
});

const DOCUMENT: DocumentWithPartsResponse = {
  id: "doc-1",
  project_id: "project-1",
  name: "Grec 1360",
  workflow: "draft",
  created_at: "2026-06-16T10:00:00Z",
  updated_at: "2026-06-16T10:00:00Z",
  part_count: 2,
  parts: [
    {
      id: "part-2",
      document_id: "doc-1",
      order: 1,
      image_url: "/media/parts/part-2",
      width: 800,
      height: 1000,
      reviewed: false,
      published: true,
      created_at: "2026-06-16T10:00:00Z",
    },
    {
      id: "part-1",
      document_id: "doc-1",
      order: 0,
      image_url: "/media/parts/part-1",
      width: 640,
      height: 900,
      reviewed: false,
      published: true,
      created_at: "2026-06-16T10:00:00Z",
    },
  ],
};

/**
 * The document, stored the way the platform stores it: a PATCH changes it and
 * every later GET reflects the change.
 *
 * A canned `getDocument` would hide the thing this page has to get right. The
 * panel patches its own view and then declares the write, which refetches; if
 * the refetch answered with the pre-write document forever, a mutation that
 * invalidated nothing would look identical to one that did.
 */
function seedDocument() {
  let stored: DocumentWithPartsResponse = DOCUMENT;
  vi.mocked(api.getDocument).mockImplementation(async () => stored);
  vi.mocked(api.updateDocument).mockImplementation(
    async (
      _projectId: string,
      _documentId: string,
      patch: {
        name?: string;
        workflow?: DocumentWithPartsResponse["workflow"];
      },
    ): Promise<DocumentResponse> => {
      stored = { ...stored, ...patch };
      // The PATCH answers without the parts, as the platform does.
      return {
        id: stored.id,
        project_id: stored.project_id,
        name: stored.name,
        workflow: stored.workflow,
        part_count: stored.part_count,
        created_at: stored.created_at,
        updated_at: stored.updated_at,
      };
    },
  );
}

function renderDocumentPage(
  initialPath = "/projects/project-1/documents/doc-1",
) {
  window.history.replaceState({}, "", initialPath);
  return render(<DocumentDetailPage />);
}

describe("DocumentDetailPage", () => {
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
      owner_id: "user-1",
      guidelines: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      document_count: 1,
    });
    seedDocument();
  });

  it("lists document parts in order and opens the page editor when a row is clicked", async () => {
    renderDocumentPage();

    await screen.findByRole("heading", { name: "Grec 1360" });

    expect(screen.getByAltText("Part 1")).toBeTruthy();
    expect(screen.getByAltText("Part 2")).toBeTruthy();

    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(2);

    fireEvent.click(rows[0]);
    await waitFor(() => {
      expect(testRouter().push).toHaveBeenCalledWith(
        "/projects/project-1/documents/doc-1/parts/part-1",
      );
    });
  });

  it("shows review status on each part and lets a project member mark a part reviewed", async () => {
    vi.mocked(api.updatePartReviewStatus).mockResolvedValue({
      ...DOCUMENT.parts[1],
      reviewed: true,
    });

    renderDocumentPage();

    await screen.findByRole("heading", { name: "Grec 1360" });
    expect(screen.getAllByText("unreviewed")).toHaveLength(2);

    fireEvent.click(
      screen.getByRole("button", { name: /mark part 1 reviewed/i }),
    );

    await waitFor(() => {
      expect(api.updatePartReviewStatus).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        { reviewed: true },
      );
    });
  });

  it("hides one page from the public reader without touching the others", async () => {
    vi.mocked(api.updatePartsPublished).mockResolvedValue([
      { ...DOCUMENT.parts[1], published: false },
    ]);

    renderDocumentPage();

    await screen.findByRole("heading", { name: "Grec 1360" });
    expect(screen.getByText(/2 of 2 shown publicly/)).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", {
        name: /hide part 1 from the public page/i,
      }),
    );

    // One request carrying only the page that changed. Sending the whole list
    // would let an unrelated stale row overwrite a flag someone else just set.
    await waitFor(() => {
      expect(api.updatePartsPublished).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        { parts: [{ part_id: "part-1", published: false }] },
      );
    });
  });

  it("leaves the page shown when the publish change is rejected", async () => {
    vi.mocked(api.updatePartsPublished).mockRejectedValue(
      new ApiError("Forbidden", 403),
    );

    renderDocumentPage();

    await screen.findByRole("heading", { name: "Grec 1360" });
    fireEvent.click(
      screen.getByRole("button", {
        name: /hide part 1 from the public page/i,
      }),
    );

    await waitFor(() => {
      expect(api.updatePartsPublished).toHaveBeenCalled();
    });
    expect(screen.getAllByText("shown")).toHaveLength(2);
    expect(screen.getByText(/2 of 2 shown publicly/)).toBeInTheDocument();
  });

  it("keeps review status when the API rejects the change", async () => {
    vi.mocked(api.updatePartReviewStatus).mockRejectedValue(
      new ApiError("Forbidden", 403),
    );

    renderDocumentPage();

    await screen.findByRole("heading", { name: "Grec 1360" });
    fireEvent.click(
      screen.getByRole("button", { name: /mark part 1 reviewed/i }),
    );

    await waitFor(() => {
      expect(api.updatePartReviewStatus).toHaveBeenCalled();
    });
    expect(screen.getAllByText("unreviewed")).toHaveLength(2);
  });

  it("makes the copy of the document the page editor holds stale after a rename", async () => {
    // The editor fetches the document under its own key. This page never
    // renders it, so a rename here used to leave the editor titled with the old
    // name until the freshness window expired.
    const editorDocumentKey = ["document", "project-1", "doc-1"];
    await queryClient.fetchQuery({
      queryKey: editorDocumentKey,
      queryFn: () => Promise.resolve(DOCUMENT),
      meta: taggedMeta([resourceTags.document("project-1", "doc-1")]),
    });

    renderDocumentPage();

    fireEvent.click(
      await screen.findByRole("button", { name: /grec 1360, click to edit/i }),
    );
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "MS Or. 1445 - Genesis" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save name/i }));

    await waitFor(() => {
      expect(queryClient.getQueryState(editorDocumentKey)?.isInvalidated).toBe(
        true,
      );
    });
  });

  it("redirects to login when the session is unauthorized", async () => {
    vi.mocked(api.getDocument).mockRejectedValue(
      new ApiError("Unauthorized", 401),
    );

    renderDocumentPage();

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(
      screen.queryByText("This document is not available to your account."),
    ).toBeNull();
  });
});
