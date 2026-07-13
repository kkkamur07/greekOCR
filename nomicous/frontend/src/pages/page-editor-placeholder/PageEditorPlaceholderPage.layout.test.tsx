import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { toast } from "../../components/ui/toast";
import {
  DOCUMENT,
  enableBaselinesOnCanvas,
  flushPageEditorEffects,
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

  it("renders part layout blocks and line baselines in layout edit mode", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: "line-1",
        part_id: "part-1",
        block_id: "block-1",
        order: 0,
        kind: "polygon",
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: [
          [60, 140],
          [300, 150],
        ],
        mask: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        source: "kraken",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [
        {
          id: "block-1",
          box: [
            [40, 60],
            [320, 60],
            [320, 220],
            [40, 220],
          ],
          manual_geometry: false,
        },
      ],
      lines: [
        {
          id: "line-1",
          block_id: "block-1",
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: false,
        },
      ],
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    expect(
      await screen.findByRole("heading", { name: /layout edit/i }),
    ).toBeTruthy();
    expect(screen.getByLabelText("Block block-1")).toBeTruthy();
    expect(screen.getByLabelText("Line line-1 baseline")).toBeTruthy();
  });

  it("renders layout geometry when API returns box and baseline objects", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: "line-1",
        part_id: "part-1",
        block_id: "block-1",
        order: 0,
        kind: "polygon",
        points: [
          [40, 60],
          [320, 60],
          [320, 220],
          [40, 220],
        ],
        baseline: {
          points: [
            [60, 140],
            [300, 150],
          ],
        },
        mask: null,
        source: "kraken",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [
        {
          id: "block-1",
          box: {
            points: [
              [40, 60],
              [320, 60],
              [320, 220],
              [40, 220],
            ],
          },
          manual_geometry: false,
        },
      ],
      lines: [
        {
          id: "line-1",
          block_id: "block-1",
          baseline: {
            points: [
              [60, 140],
              [300, 150],
            ],
          },
          manual_geometry: false,
        },
      ],
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    const baseline = await screen.findByLabelText("Line line-1 baseline");
    expect(baseline.getAttribute("points")).toBe("60,140 300,150");
  });
  it("edits a Line baseline and saves it as manual geometry", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: "line-1",
        part_id: "part-1",
        block_id: null,
        order: 0,
        kind: "polygon",
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: {
          points: [
            [60, 140],
            [300, 150],
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
        source: "kraken",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: "line-1",
          baseline: {
            points: [
              [60, 140],
              [300, 150],
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
          manual_geometry: false,
        },
      ],
    });
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
      {
        id: "line-1",
        part_id: "part-1",
        block_id: null,
        order: 0,
        kind: "polygon",
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: [
          [60, 140],
          [300, 150],
        ],
        mask: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        source: "kraken",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: "line-1",
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: true,
        },
      ],
    });
    mockedApi.resetPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: "line-1",
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: false,
        },
      ],
    });

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
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: "line-1",
        part_id: "part-1",
        block_id: null,
        order: 0,
        kind: "polygon",
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: [
          [60, 140],
          [300, 150],
        ],
        mask: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        source: "kraken",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: "line-1",
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: false,
        },
      ],
    });
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
});
