/**
 * The gap this closes: `runAutoSegment` and `runSegmentOcr`/`runPageOcr`
 * already reload the page after their job finishes, but that reload is the
 * continuation of one promise held by the one component instance whose
 * button was clicked. If that continuation never runs against a live
 * instance - the tab was backgrounded and its timers throttled, the
 * component remounted, or the researcher navigated away and back mid-job -
 * nothing else re-syncs, and only a hard reload recovers.
 *
 * `usePageEditorData` closes it from the other side: it listens for any job
 * finishing for the part it currently has open, independent of who started
 * that job or whether their own promise continuation is still around.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { JobCompletionEvent } from "../../../context/BackgroundJobsContext";
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

let completionListener: ((event: JobCompletionEvent) => void) | null = null;
const subscribeToJobCompletion = vi.fn(
  (listener: (event: JobCompletionEvent) => void) => {
    completionListener = listener;
    return () => {
      if (completionListener === listener) completionListener = null;
    };
  },
);

vi.mock("../../../context/BackgroundJobsContext", () => ({
  useBackgroundJobs: () => ({ subscribeToJobCompletion }),
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

function announce(event: JobCompletionEvent) {
  completionListener?.(event);
}

describe("usePageEditorData job-completion refresh", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    completionListener = null;
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    listTranscriptions.mockResolvedValue([]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([]);
    resolvePartModelBinding.mockRejectedValue(new Error("no binding"));
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("re-fetches the open part's content when a job finishes for it, and ignores a job for a different part", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(listPartLines).toHaveBeenCalledTimes(1);
    expect(listPartLines).toHaveBeenLastCalledWith(
      "project-1",
      "document-1",
      "part-1",
    );
    expect(subscribeToJobCompletion).toHaveBeenCalledTimes(1);

    // A segmentation that finished for the *other* part on this document is
    // not this mounted instance's business.
    await act(async () => {
      announce({
        jobId: "job-other-part",
        kind: "segmentation",
        documentPartId: "part-2",
        status: "done",
      });
    });
    expect(listPartLines).toHaveBeenCalledTimes(1);

    // A job for the part actually open re-syncs it - this is the whole point:
    // no promise continuation from whoever started the job had to run.
    const newLine = {
      id: "line-1",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "machine",
      manual_geometry: false,
      line_transcriptions: [],
    };
    listPartLines.mockResolvedValueOnce([newLine]);

    await act(async () => {
      announce({
        jobId: "job-this-part",
        kind: "segmentation",
        documentPartId: "part-1",
        status: "done",
      });
    });

    await waitFor(() => expect(result.current.lines).toEqual([newLine]));
    expect(listPartLines).toHaveBeenCalledTimes(2);
  });

  it("ignores a job announcement for a failed or cancelled run", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(listPartLines).toHaveBeenCalledTimes(1);

    await act(async () => {
      announce({
        jobId: "job-failed",
        kind: "transcription-page",
        documentPartId: "part-1",
        status: "failed",
      });
    });

    expect(listPartLines).toHaveBeenCalledTimes(1);
  });

  it("does not let a slow first load overwrite a refresh that already landed", async () => {
    // Opening a page whose job is about to finish runs two reads of the same
    // part at once. The refresh is the newer of the two by definition, so the
    // first load losing the race has to mean it stays quiet: landing its own
    // response afterwards would put the page back on pre-job state, with no
    // second announcement coming to correct it.
    const staleLine = {
      id: "line-stale",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "machine",
      manual_geometry: false,
      line_transcriptions: [],
    };
    const freshLine = { ...staleLine, id: "line-fresh" };

    let releaseFirstLoad = () => {};
    listPartLines.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseFirstLoad = () => resolve([staleLine]);
        }),
    );

    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(listPartLines).toHaveBeenCalledTimes(1));

    listPartLines.mockResolvedValueOnce([freshLine]);
    await act(async () => {
      announce({
        jobId: "job-mid-load",
        kind: "segmentation",
        documentPartId: "part-1",
        status: "done",
      });
    });
    await waitFor(() => expect(result.current.lines).toEqual([freshLine]));

    await act(async () => {
      releaseFirstLoad();
      await Promise.resolve();
    });

    expect(result.current.lines).toEqual([freshLine]);
  });
});
