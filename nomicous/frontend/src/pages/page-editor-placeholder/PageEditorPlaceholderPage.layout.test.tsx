import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { toast } from "../../components/ui/toast";
import {
  DOCUMENT,
  baselinePoints,
  enableBaselinesOnCanvas,
  flushPageEditorEffects,
  layoutLine,
  layoutWith,
  maskPoints,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

describe("PageEditorPlaceholderPage layout", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("edits a Line baseline and saves it as manual geometry", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      layoutLine({
        baseline: { points: baselinePoints() },
        mask: { points: maskPoints() },
      }),
    ]);
    mockedApi.getPartLayout.mockResolvedValue(
      layoutWith({
        baseline: { points: baselinePoints() },
        mask: { points: maskPoints() },
      }),
    );
    mockedApi.updateLineGeometry.mockResolvedValue({
      id: "line-1",
      baseline: [
        [60, 145],
        [300, 155],
      ],
      manual_geometry: true,
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText("Line line-1 baseline"));
    fireEvent.click(
      screen.getByRole("button", { name: /move baseline down/i }),
    );
    const toastSuccess = vi.spyOn(toast, "success");
    fireEvent.click(screen.getByRole("button", { name: /save layout/i }));

    await waitFor(() => {
      expect(mockedApi.updateLineGeometry).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "line-1",
        {
          baseline: {
            points: [
              [60, 145],
              [300, 155],
            ],
          },
          mask: {
            points: [
              [55, 110],
              [305, 118],
              [300, 178],
              [50, 168],
            ],
          },
        },
      );
    });
    await waitFor(() => {
      expect(toastSuccess).toHaveBeenCalledWith("Manual geometry saved");
    });
  });

  it("resets selected Line layout through the API and refreshes the canvas state", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      layoutLine({ manual_geometry: true }),
    ]);
    mockedApi.getPartLayout.mockResolvedValue(
      layoutWith({ manual_geometry: true }),
    );
    mockedApi.resetPartLayout.mockResolvedValue(layoutWith());

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText("Line line-1 baseline"));
    const toastSuccess = vi.spyOn(toast, "success");
    fireEvent.click(screen.getByRole("button", { name: /reset layout/i }));

    await waitFor(() => {
      expect(mockedApi.resetPartLayout).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        { line_ids: ["line-1"] },
      );
    });
    await waitFor(() => {
      expect(toastSuccess).toHaveBeenCalledWith("Layout reset");
    });
  });

  it("shows a member-only error when the layout save API rejects access", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([layoutLine()]);
    mockedApi.getPartLayout.mockResolvedValue(layoutWith());
    mockedApi.updateLineGeometry.mockRejectedValue(
      new ApiError("Forbidden", 403),
    );

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText("Line line-1 baseline"));
    fireEvent.click(
      screen.getByRole("button", { name: /move baseline down/i }),
    );
    fireEvent.click(screen.getByRole("button", { name: /save layout/i }));

    expect(
      await screen.findByText("Only project members can edit layout."),
    ).toBeTruthy();
    expect(
      screen.getByLabelText("Line line-1 baseline").getAttribute("points"),
    ).toBe("60,140 300,150");
  });

  it("shows a member-only error when the layout reset API rejects access", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      layoutLine({ manual_geometry: true }),
    ]);
    mockedApi.getPartLayout.mockResolvedValue(
      layoutWith({ manual_geometry: true }),
    );
    mockedApi.resetPartLayout.mockRejectedValue(new ApiError("Forbidden", 403));

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText("Line line-1 baseline"));
    const toastSuccess = vi.spyOn(toast, "success");
    fireEvent.click(screen.getByRole("button", { name: /reset layout/i }));

    // Both call sites drop the returned promise, so this rejection used to
    // leave the page saying nothing whatsoever.
    expect(
      await screen.findByText(
        "Segment API unavailable: Only project members can edit layout.",
      ),
    ).toBeTruthy();
    expect(toastSuccess).not.toHaveBeenCalledWith("Layout reset");
  });
});
