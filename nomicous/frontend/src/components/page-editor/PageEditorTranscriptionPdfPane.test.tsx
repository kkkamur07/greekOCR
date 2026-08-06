import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { PageEditorTranscriptionPdfPane } from "./PageEditorTranscriptionPdfPane";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      generateTranscriptionPdf: vi.fn(),
    },
  };
});

const mockedGenerateTranscriptionPdf =
  api.generateTranscriptionPdf as ReturnType<typeof vi.fn>;

describe("PageEditorTranscriptionPdfPane", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGenerateTranscriptionPdf.mockResolvedValue(
      new Blob(["%PDF"], { type: "application/pdf" }),
    );
  });

  it("loads the transcription PDF into an embedded preview", async () => {
    const createObjectURL = vi.fn(() => "blob:preview");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTitle("Transcription PDF preview")).toHaveAttribute(
        "src",
        "blob:preview",
      );
    });
    expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledWith(
      "project-1",
      "doc-1",
      "part-1",
    );

    vi.unstubAllGlobals();
  });

  it("embeds with an iframe, never an object that object-src 'none' blocks", async () => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:preview"),
      revokeObjectURL: vi.fn(),
    });

    const { container } = render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    const embed = await screen.findByTitle("Transcription PDF preview");
    expect(embed.tagName).toBe("IFRAME");
    expect(container.querySelector("object")).toBeNull();
    expect(container.querySelector("embed")).toBeNull();

    vi.unstubAllGlobals();
  });

  it("offers an escape hatch outside the embed once the blob exists", async () => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:preview"),
      revokeObjectURL: vi.fn(),
    });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    // The embed cannot report its own failure, so the way out must not be
    // fallback content inside it.
    const openLink = await screen.findByRole("link", {
      name: /open the pdf in a new tab/i,
    });
    expect(openLink).toHaveAttribute("href", "blob:preview");
    expect(openLink).toHaveAttribute("target", "_blank");
    expect(openLink.closest("iframe")).toBeNull();

    const download = screen.getByRole("button", { name: /download pdf/i });
    expect(download).toBeEnabled();
    expect(download.closest("iframe")).toBeNull();

    vi.unstubAllGlobals();
  });

  it("keeps Download reachable while the preview is still loading", () => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:preview"),
      revokeObjectURL: vi.fn(),
    });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /download pdf/i })).toBeEnabled();

    vi.unstubAllGlobals();
  });

  it("refetches when refreshKey changes", async () => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:preview"),
      revokeObjectURL: vi.fn(),
    });

    const { rerender } = render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledTimes(1);
    });

    rerender(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={2}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledTimes(2);
    });

    vi.unstubAllGlobals();
  });

  it("downloads the PDF when the user clicks Download", async () => {
    const blob = new Blob(["%PDF"], { type: "application/pdf" });
    mockedGenerateTranscriptionPdf.mockResolvedValue(blob);
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:download"),
      revokeObjectURL: vi.fn(),
    });

    // Spy on the click rather than on document.createElement: the pane also
    // renders a real <a> escape hatch, and a createElement stub that returns a
    // plain object for every "a" breaks React's own rendering.
    const clicked: { href: string | null; download: string }[] = [];
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        clicked.push({
          href: this.getAttribute("href"),
          download: this.download,
        });
      });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /download pdf/i }),
    );

    await waitFor(() => {
      expect(mockedGenerateTranscriptionPdf).toHaveBeenCalledTimes(2);
      expect(clicked).toEqual([
        { href: "blob:download", download: "page-1_transcription.pdf" },
      ]);
    });

    clickSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it("shows API errors from transcription PDF generation", async () => {
    mockedGenerateTranscriptionPdf.mockRejectedValue(
      new ApiError("Forbidden", 403),
    );
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(),
      revokeObjectURL: vi.fn(),
    });

    render(
      <PageEditorTranscriptionPdfPane
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-1_transcription.pdf"
        refreshKey={1}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText("Forbidden")).toBeTruthy();

    vi.unstubAllGlobals();
  });
});
