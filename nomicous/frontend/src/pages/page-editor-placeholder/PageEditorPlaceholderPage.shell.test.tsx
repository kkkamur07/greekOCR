import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ApiError } from "../../api/errors";
import { queryClient, taggedMeta } from "../../api/queryClient";
import { resourceTags } from "../../api/resources";
import {
  DOCUMENT,
  flushPageEditorEffects,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

describe("PageEditorPlaceholderPage shell", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("does not render protected media when the API rejects access", async () => {
    mockedApi.getDocument.mockRejectedValue(new ApiError("Forbidden", 403));

    renderPageEditor();

    expect(await screen.findByText("Page unavailable")).toBeTruthy();
    expect(
      screen.getByText("This page is not available to your account."),
    ).toBeTruthy();
    expect(screen.queryByAltText("Page 1")).toBeNull();
  });

  it("makes the reads that copy the workflow stale when it publishes", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    // The public page, as a reader holds it. The editor never renders it, so
    // publishing used to leave it on "not published" until it expired.
    const publicDocumentKey = ["public-document", "project-1", "doc-1"];
    await queryClient.fetchQuery({
      queryKey: publicDocumentKey,
      queryFn: () => Promise.resolve({ workflow: "draft" }),
      meta: taggedMeta([resourceTags.publicDocument("project-1", "doc-1")]),
    });

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^process/i }));
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /publish live page/i }),
    );

    await waitFor(() => {
      expect(queryClient.getQueryState(publicDocumentKey)?.isInvalidated).toBe(
        true,
      );
    });
  });
});
