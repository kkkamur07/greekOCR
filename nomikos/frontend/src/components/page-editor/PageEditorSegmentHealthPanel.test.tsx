/**
 * The panel's job is to refuse to be a one-click cleanup tool.
 *
 * Every assertion here is about something the panel declines to do on its own:
 * delete on a single click, offer a trim that would halve a duplicated line, or
 * report page-relative findings without saying when the page size was guessed.
 * The geometry is tested in Python; what is left to get wrong is the interface.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { SegmentHealthResponse } from "../../api/client";
import { PageEditorSegmentHealthPanel } from "./PageEditorSegmentHealthPanel";

const UPPER = "11111111-1111-4111-8111-111111111111";
const LOWER = "22222222-2222-4222-8222-222222222222";
const SPECK = "33333333-3333-4333-8333-333333333333";
const PRIMARY = "44444444-4444-4444-8444-444444444444";
const FRAGMENT = "55555555-5555-4555-8555-555555555555";

function report(
  overrides: Partial<SegmentHealthResponse> = {},
): SegmentHealthResponse {
  return {
    part_id: "00000000-0000-4000-8000-000000000000",
    page_width: 2479,
    page_height: 3508,
    measured_page: true,
    line_count: 40,
    considered_count: 40,
    finding_count: 0,
    suspects: [],
    spanning: [],
    fragments: [],
    overlaps: [],
    ...overrides,
  } as SegmentHealthResponse;
}

function panel(
  props: Partial<Parameters<typeof PageEditorSegmentHealthPanel>[0]>,
) {
  return render(
    <PageEditorSegmentHealthPanel
      report={report()}
      loading={false}
      error={null}
      pending={null}
      onApply={() => {}}
      onRefresh={() => {}}
      {...props}
    />,
  );
}

describe("PageEditorSegmentHealthPanel", () => {
  it("needs two clicks before it will delete a suspect", () => {
    const onApply = vi.fn();
    panel({
      report: report({
        finding_count: 1,
        suspects: [
          { line_id: SPECK, reasons: ["narrower than 6% of the page"] },
        ],
      }),
      onApply,
    });

    fireEvent.click(screen.getByRole("button", { name: "Delete…" }));
    expect(onApply).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Really delete it" }));
    expect(onApply).toHaveBeenCalledWith({ kind: "delete", lineId: SPECK });
  });

  it("shows the reasons a segment was flagged, not just that it was", () => {
    panel({
      report: report({
        finding_count: 1,
        suspects: [
          {
            line_id: SPECK,
            reasons: ["narrower than 6% of the page", "outside every column"],
          },
        ],
      }),
    });

    expect(
      screen.getByText(/narrower than 6% of the page; outside every column/),
    ).toBeTruthy();
  });

  it("offers no trim for two lines that are one line drawn twice", () => {
    panel({
      report: report({
        finding_count: 1,
        overlaps: [
          {
            upper_id: UPPER,
            lower_id: LOWER,
            ratio: 0.82,
            cut: 1200,
            upper_loss: 0,
            lower_loss: 0,
            duplicate: true,
          },
        ],
      }),
    });

    expect(screen.queryByRole("button", { name: /Trim apart/ })).toBeNull();
    expect(screen.getByText(/one line drawn twice/)).toBeTruthy();
  });

  it("puts the cost of a trim on screen before anyone agrees to it", () => {
    panel({
      report: report({
        finding_count: 1,
        overlaps: [
          {
            upper_id: UPPER,
            lower_id: LOWER,
            ratio: 0.28,
            cut: 1200,
            upper_loss: 0.12,
            lower_loss: 0.07,
            duplicate: false,
          },
        ],
      }),
    });

    expect(screen.getByText(/Trimming costs 12% and 7%/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Trim apart" })).toBeTruthy();
  });

  it("says which row survives a merge, because that is where the text is", () => {
    const onApply = vi.fn();
    panel({
      report: report({
        finding_count: 1,
        fragments: [{ primary_id: PRIMARY, fragment_id: FRAGMENT }],
      }),
      onApply,
    });

    expect(screen.getByText(/so its transcription survives/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Merge" }));
    expect(onApply).toHaveBeenCalledWith({
      kind: "merge",
      primaryId: PRIMARY,
      fragmentId: FRAGMENT,
    });
  });

  it("admits when the page size was estimated rather than read", () => {
    panel({ report: report({ measured_page: false }) });
    expect(
      screen.getByText(/page size estimated from the segments/),
    ).toBeTruthy();
  });

  it("says nothing was found rather than showing an empty panel", () => {
    panel({ report: report() });
    expect(
      screen.getByText(/Nothing systematic found across 40 segments/),
    ).toBeTruthy();
  });

  it("disables only the row being applied", () => {
    panel({
      report: report({
        finding_count: 2,
        spanning: [
          { line_id: UPPER, cuts: [1240], piece_count: 2 },
          { line_id: LOWER, cuts: [1240], piece_count: 2 },
        ],
      }),
      pending: UPPER,
    });

    const buttons = screen.getAllByRole("button");
    const cutting = buttons.find((b) => b.textContent === "Cutting…");
    const idle = buttons.find((b) => b.textContent === "Cut at the gutter");
    expect((cutting as HTMLButtonElement).disabled).toBe(true);
    expect((idle as HTMLButtonElement).disabled).toBe(false);
  });

  it("keeps the findings on screen when a fix is refused", () => {
    // The refusal arrives with a re-read behind it, so the list underneath is
    // the reviewer's evidence for what happened. Replacing it with "Try again"
    // would throw that away at the one moment it is worth reading.
    panel({
      error: "This segment is no longer offered a column split",
      report: report({
        finding_count: 1,
        spanning: [{ line_id: UPPER, cuts: [1240], piece_count: 2 }],
      }),
    });

    expect(screen.getByRole("alert").textContent).toContain(
      "no longer offered",
    );
    expect(
      screen.getByRole("button", { name: "Cut at the gutter" }),
    ).toBeTruthy();
  });

  it("shows only the error when there is no report to keep", () => {
    panel({ error: "Could not check this page.", report: null });

    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.queryByText(/Nothing systematic found/)).toBeNull();
  });
});
