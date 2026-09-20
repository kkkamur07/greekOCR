import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { clearAccessToken, setAccessToken } from "../../../auth/storage";
import { usePageEditorData } from "./usePageEditorData";

const getDocument = vi.fn();
const getPartLayout = vi.fn();
const listPartLines = vi.fn();
const listTranscriptions = vi.fn();
const getPagePairing = vi.fn();
const listInferenceModels = vi.fn();
const resolvePartModelBinding = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    getDocument: (...args: unknown[]) => getDocument(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    listTranscriptions: (...args: unknown[]) => listTranscriptions(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    resolvePartModelBinding: (...args: unknown[]) =>
      resolvePartModelBinding(...args),
  },
}));

vi.mock("../../../context/BackgroundJobsContext", () => ({
  useBackgroundJobs: () => ({ subscribeToJobCompletion: () => () => {} }),
}));

const DOCUMENT = {
  id: "document-1",
  project_id: "project-1",
  parts: [
    { id: "part-1", order: 0 },
    { id: "part-2", order: 1 },
  ],
};

const EMPTY_PAIRING = {
  text_lines: [],
  pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
};

const TRANSCRIBE_MODEL = { id: "htr-1", task: "transcribe", name: "htr" };
const SEGMENT_A = { id: "seg-a", task: "segment", name: "blla-segment" };
const SEGMENT_B = { id: "seg-b", task: "segment", name: "pp-ocr" };

describe("usePageEditorData segment models", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    listTranscriptions.mockResolvedValue([]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([
      TRANSCRIBE_MODEL,
      SEGMENT_A,
      SEGMENT_B,
      { id: "bin-1", task: "binarize", name: "bin" },
    ]);
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: TRANSCRIBE_MODEL });
        return Promise.reject(new Error("no binding"));
      },
    );
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("filters segment models from the single catalog call", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(listInferenceModels).toHaveBeenCalledTimes(1);
    expect(result.current.segmentModels.map((model) => model.id)).toEqual([
      "seg-a",
      "seg-b",
    ]);
    expect(result.current.transcribeModels.map((model) => model.id)).toEqual([
      "htr-1",
    ]);
    expect(resolvePartModelBinding).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      "segment",
    );
  });

  it("leaves Default selected when the segment binding rejects, without an error", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBeNull();
    expect(result.current.error).toBeNull();
    // The HTR picker still falls back to the first catalog row.
    expect(result.current.selectedTranscribeModelId).toBe("htr-1");
  });

  it("preselects the resolved segment binding", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "segment") return Promise.resolve({ model: SEGMENT_B });
        return Promise.reject(new Error("no binding"));
      },
    );

    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-b");
  });

  it("keeps the explicit choice across a page turn without refetching the catalog", async () => {
    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedSegmentModelId).toBeNull();

    act(() => {
      result.current.setSelectedSegmentModelId("seg-a");
    });
    expect(result.current.selectedSegmentModelId).toBe("seg-a");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-a");
    expect(listInferenceModels).toHaveBeenCalledTimes(1);
    expect(result.current.segmentModels.map((model) => model.id)).toEqual([
      "seg-a",
      "seg-b",
    ]);
  });
});
