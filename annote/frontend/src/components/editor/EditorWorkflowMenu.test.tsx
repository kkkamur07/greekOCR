import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import EditorWorkflowMenu from "./EditorWorkflowMenu";

const baseProps = {
  locked: false,
  binarized: false,
  binarizing: false,
  segmenting: false,
  ocrRunning: false,
  exporting: false,
  hasSegments: true,
  pdfPanelOpen: false,
  pdfPanelMode: null as const,
  onBinarize: vi.fn(),
  onClearBinarize: vi.fn(),
  onAutoSegment: vi.fn(),
  onOcrPage: vi.fn(),
  onPdfOpen: vi.fn(),
  onPdfClose: vi.fn(),
};

describe("EditorWorkflowMenu", () => {
  it("opens live PDF preview from the workflow menu", () => {
    const onPdfOpen = vi.fn();
    render(<EditorWorkflowMenu {...baseProps} onPdfOpen={onPdfOpen} />);

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /transcription pdf/i }));
    expect(onPdfOpen).toHaveBeenCalledWith("preview");
  });

  it("disables share until the page is locked", () => {
    const onPdfOpen = vi.fn();
    render(<EditorWorkflowMenu {...baseProps} onPdfOpen={onPdfOpen} />);

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    const share = screen.getByRole("menuitem", { name: /share pdf/i });
    expect(share).toBeDisabled();
    fireEvent.click(share);
    expect(onPdfOpen).not.toHaveBeenCalled();
  });

  it("opens share mode when locked", () => {
    const onPdfOpen = vi.fn();
    render(<EditorWorkflowMenu {...baseProps} locked onPdfOpen={onPdfOpen} />);

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /share pdf/i }));
    expect(onPdfOpen).toHaveBeenCalledWith("share");
  });

  it("closes the PDF panel when preview is already open", () => {
    const onPdfClose = vi.fn();
    render(
      <EditorWorkflowMenu
        {...baseProps}
        pdfPanelOpen
        pdfPanelMode="preview"
        onPdfClose={onPdfClose}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /transcription pdf/i }));
    expect(onPdfClose).toHaveBeenCalled();
  });
});
