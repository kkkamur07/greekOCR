import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import TranscriptionPdfMenu from "./TranscriptionPdfMenu";

describe("TranscriptionPdfMenu", () => {
  it("opens live preview from the primary button", () => {
    const onOpen = vi.fn();
    render(
      <TranscriptionPdfMenu
        locked={false}
        panelOpen={false}
        panelMode={null}
        onOpen={onOpen}
        onClose={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /transcription pdf/i }));
    expect(onOpen).toHaveBeenCalledWith("preview");
  });

  it("closes the panel when preview is already open", () => {
    const onClose = vi.fn();
    render(
      <TranscriptionPdfMenu
        locked={false}
        panelOpen
        panelMode="preview"
        onOpen={vi.fn()}
        onClose={onClose}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /transcription pdf/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("disables share until the page is locked", () => {
    const onOpen = vi.fn();
    render(
      <TranscriptionPdfMenu
        locked={false}
        panelOpen={false}
        panelMode={null}
        onOpen={onOpen}
        onClose={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "▾" }));
    const share = screen.getByRole("menuitem", { name: /share/i });
    expect(share).toBeDisabled();
    fireEvent.click(share);
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("opens share mode when locked", () => {
    const onOpen = vi.fn();
    render(
      <TranscriptionPdfMenu
        locked
        panelOpen={false}
        panelMode={null}
        onOpen={onOpen}
        onClose={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "▾" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /share/i }));
    expect(onOpen).toHaveBeenCalledWith("share");
  });
});
