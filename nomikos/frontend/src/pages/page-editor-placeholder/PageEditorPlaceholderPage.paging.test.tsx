import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  documentWithPages,
  flushPageEditorEffects,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

/** The editor route for a page, as the address bar should end up holding it. */
function partHref(partId: string): string {
  return `/projects/project-1/documents/doc-1/parts/${partId}`;
}

/**
 * Wait for the toolbar to say the editor is on page `pageNumber`.
 *
 * The assertion is on the heading's text rather than on its accessible name:
 * the page number is a separate text node, and the accessible-name algorithm
 * closes the gap between it and the document title, so the name a screen
 * reader hears is not the string the researcher reads.
 */
async function expectEditorOnPage(pageNumber: number): Promise<void> {
  await waitFor(() => {
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      new RegExp(`^Grec 1360 · p\\.${pageNumber}$`),
    );
  });
}

describe("PageEditorPlaceholderPage paging", () => {
  // Paging moves the URL with window.history.pushState rather than a router
  // navigation (see useDocumentPaging), so the assertions watch the history
  // API instead of the router mock.
  let pushState: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    pushState = vi.spyOn(window.history, "pushState");
    resetPageEditorApiMocks();
    // Whether the rail is open is remembered across sessions, and the stubbed
    // localStorage outlives a test. Clear it so one test hiding the rail is
    // not the next test's starting state.
    localStorage.clear();
  });

  afterEach(async () => {
    pushState.mockRestore();
    await flushPageEditorEffects();
  });

  it("lists every page in the rail without fetching every thumbnail", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(40));

    renderPageEditor();
    await expectEditorOnPage(1);

    // Every page is reachable: the rail is the document, not a window on it.
    expect(screen.getAllByRole("button", { name: /^Page \d+$/ })).toHaveLength(
      40,
    );

    // A thumbnail is only mounted for the rows the rail believes are on
    // screen. With no layout to measure, that is the active page and six
    // either side of it, so page 1 loads seven thumbnails and not forty.
    expect(screen.getAllByAltText(/^Page \d+ thumbnail$/)).toHaveLength(7);
  });

  it("turns the page from the rail without leaving the editor", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(3));

    renderPageEditor();
    await expectEditorOnPage(1);

    const pageThree = screen.getByRole("button", { name: "Page 3" });
    pageThree.focus();
    fireEvent.click(pageThree);

    await waitFor(() => {
      expect(mockedApi.listPartLines).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-3",
      );
    });

    // The header indicator follows the page the editor is actually showing.
    await expectEditorOnPage(3);

    // A reload or a shared link has to land back here.
    expect(pushState).toHaveBeenCalledWith(null, "", partHref("part-3"));

    // The editor was not torn down and rebuilt around the new page: the
    // control that was pressed is still mounted and still holds focus. A page
    // turn that went through the route instead would leave this on the body.
    expect(globalThis.document.activeElement).toBe(pageThree);
  });

  it("moves a page at a time from the keyboard", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(3));

    renderPageEditor();
    await expectEditorOnPage(1);

    fireEvent.keyDown(window, { key: "PageDown" });

    await expectEditorOnPage(2);
    expect(pushState).toHaveBeenLastCalledWith(null, "", partHref("part-2"));

    fireEvent.keyDown(window, { key: "PageUp" });

    await expectEditorOnPage(1);
    expect(pushState).toHaveBeenLastCalledWith(null, "", partHref("part-1"));
  });

  it("does not page past either end of the document", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(2));

    renderPageEditor();
    await expectEditorOnPage(1);

    expect(
      screen.getByRole("button", { name: "Previous page" }),
    ).toBeDisabled();
    const next = screen.getByRole("button", { name: "Next page" });
    expect(next).toBeEnabled();

    fireEvent.click(next);

    await expectEditorOnPage(2);
    expect(screen.getByRole("button", { name: "Previous page" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();

    // The last page is the last page: PageDown on it is not a route change.
    pushState.mockClear();
    fireEvent.keyDown(window, { key: "PageDown" });
    await flushPageEditorEffects();
    expect(pushState).not.toHaveBeenCalled();
  });

  it("adopts the page the address bar moves to on its own", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(3));

    const view = renderPageEditor();
    await expectEditorOnPage(1);

    fireEvent.click(screen.getByRole("button", { name: "Page 2" }));
    await expectEditorOnPage(2);

    // The stubbed router records a push without moving the address bar, so the
    // route landing on page 2 is played back by hand. The editor has to sit
    // still through its own push: this is where a naive "adopt the route"
    // would re-run the page load it just did.
    window.history.replaceState({}, "", partHref("part-2"));
    view.rerenderFromUrl();
    await expectEditorOnPage(2);

    // Browser Back: now the route moves without the editor asking it to, and
    // the editor has to follow rather than keep showing the page it chose.
    window.history.replaceState({}, "", partHref("part-1"));
    view.rerenderFromUrl();

    await expectEditorOnPage(1);
    await waitFor(() => {
      expect(mockedApi.listPartLines).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
      );
    });
  });

  it("walks the rail with the arrow keys and leaves Space to the button", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(3));

    renderPageEditor();
    await expectEditorOnPage(1);

    const pageOne = screen.getByRole("button", { name: "Page 1" });
    pageOne.focus();

    // The editor binds the bare arrow keys for nudging a selected segment. In
    // the rail they move along the pages instead, and moving focus is not a
    // page turn: the canvas is still showing page 1.
    fireEvent.keyDown(pageOne, { key: "ArrowDown" });
    expect(globalThis.document.activeElement).toBe(
      screen.getByRole("button", { name: "Page 2" }),
    );
    fireEvent.keyDown(globalThis.document.activeElement!, { key: "End" });
    expect(globalThis.document.activeElement).toBe(
      screen.getByRole("button", { name: "Page 3" }),
    );
    await expectEditorOnPage(1);
    expect(pushState).not.toHaveBeenCalled();

    // Space is how half the keyboard world presses a button, and the canvas
    // must not swallow it into its pan override on the way. The override is
    // observable, so this is the part jsdom can actually witness: the browser
    // is what turns the un-prevented keypress into a click.
    const host = globalThis.document.querySelector(".pe-canvas-host");
    fireEvent.keyDown(globalThis.document.activeElement!, { code: "Space" });
    expect(host?.classList.contains("pe-canvas-host--panning")).toBe(false);
  });

  it("hands focus to the edge tab when the rail is hidden", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(3));

    renderPageEditor();
    await expectEditorOnPage(1);

    const collapse = screen.getByRole("button", { name: "Hide the page list" });
    collapse.focus();
    fireEvent.click(collapse);

    await waitFor(() => {
      expect(
        screen.queryByRole("navigation", { name: "Document pages" }),
      ).toBeNull();
    });
    expect(globalThis.document.activeElement).toBe(
      screen.getByRole("button", { name: "Show the page list" }),
    );

    // And the tab brings it back, so hiding it is not a one-way door.
    fireEvent.click(screen.getByRole("button", { name: "Show the page list" }));
    expect(
      await screen.findByRole("navigation", { name: "Document pages" }),
    ).toBeTruthy();
  });

  it("offers no rail and no page turn on a one-page document", async () => {
    mockedApi.getDocument.mockResolvedValue(documentWithPages(1));

    renderPageEditor();
    await expectEditorOnPage(1);

    expect(
      screen.queryByRole("navigation", { name: "Document pages" }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Show the page list" }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: "Previous page" }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
  });
});
