import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import * as api from "@/lib/api";
import TranscriptionPdfPanel from "./TranscriptionPdfPanel";

describe("TranscriptionPdfPanel", () => {
  it("loads the live preview PDF into an embedded object", async () => {
    const createObjectURL = vi.fn(() => "blob:preview");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["%PDF"], { type: "application/pdf" })),
      }),
    );

    render(
      <TranscriptionPdfPanel
        stem="folio"
        mode="preview"
        locked={false}
        refreshKey={42}
        onClose={vi.fn()}
        onSwitchMode={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTitle("Transcription PDF preview")).toHaveAttribute("data", "blob:preview");
    });
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/pages/folio/transcription.pdf?t=42"), {
      signal: expect.any(AbortSignal),
    });

    vi.unstubAllGlobals();
  });

  it("disables share tab when unlocked", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["%PDF"], { type: "application/pdf" })),
      }),
    );

    render(
      <TranscriptionPdfPanel
        stem="folio"
        mode="preview"
        locked={false}
        refreshKey={1}
        onClose={vi.fn()}
        onSwitchMode={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Share" })).toBeDisabled();
    vi.unstubAllGlobals();
  });

  it("renders as a side panel beside the canvas", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["%PDF"], { type: "application/pdf" })),
      }),
    );

    render(
      <TranscriptionPdfPanel
        stem="folio"
        mode="preview"
        locked={false}
        refreshKey={1}
        onClose={vi.fn()}
        onSwitchMode={vi.fn()}
      />,
    );

    expect(screen.getByLabelText("Transcription PDF preview")).toHaveClass("w-1/2");
    vi.unstubAllGlobals();
  });

  it("refetches the PDF when refreshKey changes after a save", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["%PDF"], { type: "application/pdf" })),
      });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:preview"), revokeObjectURL: vi.fn() });

    const { rerender } = render(
      <TranscriptionPdfPanel
        stem="folio"
        mode="preview"
        locked={false}
        refreshKey={1}
        onClose={vi.fn()}
        onSwitchMode={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/pages/folio/transcription.pdf?t=1"),
        expect.any(Object),
      );
    });

    rerender(
      <TranscriptionPdfPanel
        stem="folio"
        mode="preview"
        locked={false}
        refreshKey={2}
        onClose={vi.fn()}
        onSwitchMode={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/pages/folio/transcription.pdf?t=2"),
        expect.any(Object),
      );
    });

    vi.unstubAllGlobals();
  });

  it("downloads only when the user clicks Download", async () => {
    const blob = new Blob(["%PDF"], { type: "application/pdf" });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(blob),
      }),
    );
    vi.spyOn(api, "fetchTranscriptionPdfBlob").mockResolvedValue(blob);
    const createObjectURL = vi.fn(() => "blob:download");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });

    const anchor = { click: vi.fn(), href: "", download: "" } as unknown as HTMLAnchorElement;
    const originalCreateElement = document.createElement.bind(document);
    const createElement = vi.spyOn(document, "createElement").mockImplementation((tagName, options) => {
      if (tagName === "a") return anchor;
      return originalCreateElement(tagName, options);
    });

    render(
      <TranscriptionPdfPanel
        stem="folio"
        mode="preview"
        locked={false}
        refreshKey={1}
        onClose={vi.fn()}
        onSwitchMode={vi.fn()}
      />,
    );

    expect(api.fetchTranscriptionPdfBlob).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    await waitFor(() => {
      expect(api.fetchTranscriptionPdfBlob).toHaveBeenCalledWith("folio", "preview");
      expect(anchor.download).toBe("folio_transcription.pdf");
      expect(anchor.click).toHaveBeenCalled();
    });

    createElement.mockRestore();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });
});
