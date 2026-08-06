import { render, screen, waitFor } from "@testing-library/react";
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
        "data",
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
