import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

function groundTruthLine(id: string, order: number, text: string) {
  return line({
    id,
    order,
    points: [
      [10, 10 + order * 50],
      [110, 10 + order * 50],
      [110, 40 + order * 50],
      [10, 40 + order * 50],
    ],
    line_transcriptions: [
      {
        id: `${id}-tx-1`,
        transcription_id: "ground-truth-1",
        transcription_kind: "ground_truth",
        text,
        confidence: null,
      },
    ],
  });
}

function twoLines() {
  return [
    groundTruthLine("line-1", 0, "first words"),
    groundTruthLine("line-2", 1, "second words"),
  ];
}

function seedSideBySide(on: boolean) {
  localStorage.setItem(
    "nomikos_page_editor_settings",
    JSON.stringify({ sideBySide: on }),
  );
}

function seedEditor() {
  mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  mockedApi.listPartLines.mockResolvedValue(twoLines());
}

describe("PageEditorPlaceholderPage side by side", () => {
  beforeEach(() => {
    localStorage.clear();
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
    localStorage.clear();
  });

  it("renders the text panel next to the canvas when the setting is on", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();

    expect(await screen.findByLabelText("Page geometry canvas")).toBeTruthy();
    expect(container.querySelector(".pe-side-by-side")).not.toBeNull();
    expect(container.querySelector(".pe-text-panel")).not.toBeNull();
  });

  it("mirrors hovering a text line onto the same canvas segment", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    const group = container.querySelector('[data-line-id="line-1"]');
    expect(group).not.toBeNull();
    fireEvent.mouseEnter(group!);

    await waitFor(() => {
      expect(
        screen
          .getByRole("button", { name: /^Segment 1/ })
          .classList.contains("is-hovered"),
      ).toBe(true);
    });
  });

  it("selects the segment when its text line is clicked", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    const group = container.querySelector('[data-line-id="line-1"]');
    expect(group).not.toBeNull();
    fireEvent.click(group!);

    expect(
      await screen.findByRole("heading", { name: /segment 1/i }),
    ).toBeTruthy();
  });

  it("saves the edited text and moves to the next segment in reading order", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    const group = container.querySelector('[data-line-id="line-1"]');
    expect(group).not.toBeNull();
    fireEvent.doubleClick(group!);

    const editor = await screen.findByDisplayValue("first words");
    fireEvent.change(editor, { target: { value: "first words edited" } });
    fireEvent.keyDown(editor, { key: "Enter", shiftKey: false });

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "first words edited" },
      );
    });
    await waitFor(() => {
      const panelEditor = container.querySelector(
        ".pe-text-line-editor",
      ) as HTMLTextAreaElement | null;
      expect(panelEditor?.value).toBe("second words");
    });
    expect(
      await screen.findByRole("heading", { name: /segment 2/i }),
    ).toBeTruthy();
  });

  it("closes the open editor when a different segment is selected", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    const group = container.querySelector('[data-line-id="line-1"]');
    expect(group).not.toBeNull();
    fireEvent.doubleClick(group!);
    expect(await screen.findByDisplayValue("first words")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /^Segment 2/ }));

    await waitFor(() => {
      expect(screen.queryByDisplayValue("first words")).toBeNull();
    });
    expect(
      await screen.findByRole("heading", { name: /segment 2/i }),
    ).toBeTruthy();
  });

  it("hides the text panel when the setting is off", async () => {
    seedSideBySide(false);
    seedEditor();
    const { container } = renderPageEditor();

    expect(await screen.findByLabelText("Page geometry canvas")).toBeTruthy();
    expect(container.querySelector(".pe-text-panel")).toBeNull();
    expect(container.querySelector(".pe-side-by-side")).toBeNull();
  });
});
