import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useLocalInferenceRuns } from "../../../inference";
import { useLayoutMutations } from "./useLayoutMutations";

const segmentPart = vi.fn();
const listPartLines = vi.fn();
const getPartLayout = vi.fn();
const getPagePairing = vi.fn();
const persistLocalSegment = vi.fn();
const runLocalInference = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    segmentPart: (...args: unknown[]) => segmentPart(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
    persistLocalSegment: (...args: unknown[]) => persistLocalSegment(...args),
  },
}));

vi.mock("../../../api/imageCache", () => ({
  fetchPartImage: async () => new Blob(["page-image"], { type: "image/png" }),
}));

vi.mock("../../../inference", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../inference")>();
  return {
    ...actual,
    runLocalInference: (...args: unknown[]) => runLocalInference(...args),
  };
});

const SEGMENT_OUTPUT = { blocks: [], lines: [] };

type TrackLocalTask = <T>(
  meta: { label: string; kind: string },
  run: (signal: AbortSignal) => Promise<T>,
) => Promise<T>;

function setup(options?: {
  trackLocalTask?: TrackLocalTask;
  cloudInferenceEnabled?: boolean;
}) {
  const setPairingError = vi.fn();
  const trackJobAndWait = vi.fn().mockResolvedValue({ status: "done" });

  const defaultTrackLocalTask: TrackLocalTask = (_meta, run) =>
    run(new AbortController().signal);

  // The real run registry, so the three abort causes are told apart exactly the
  // way the page tells them apart.
  const view = renderHook(() => {
    const { localInference, abortRunToCloud } = useLocalInferenceRuns(
      () => true,
    );
    const mutations = useLayoutMutations({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      layout: { blocks: [], lines: [] },
      setLayout: vi.fn(),
      lines: [],
      setLines: vi.fn(),
      setLineError: vi.fn(),
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError,
      selectedSegmentId: null,
      setSelectedSegmentId: vi.fn(),
      setApprovedTextDraft: vi.fn(),
      onDrawComplete: vi.fn(),
      partImageUrl: "http://localhost:8000/media/parts/part-1",
      shouldUseLocalPath: () => true,
      cloudInferenceEnabled: options?.cloudInferenceEnabled ?? true,
      segmentRegistryModelId: "blla-segment",
      localInference,
      trackJobAndWait,
      trackLocalTask: options?.trackLocalTask ?? defaultTrackLocalTask,
    });
    return { ...mutations, abortRunToCloud };
  });

  return { view, setPairingError, trackJobAndWait };
}

/** A local run that never finishes on its own - only an abort ends it. */
function helperRunBlockedUntilAbort() {
  let reachedHelper: () => void;
  const atHelper = new Promise<void>((resolve) => {
    reachedHelper = resolve;
  });
  runLocalInference.mockImplementationOnce(
    (request: { signal: AbortSignal }) =>
      new Promise((_resolve, reject) => {
        request.signal.addEventListener("abort", () => {
          reject(new DOMException("The operation was aborted.", "AbortError"));
        });
        reachedHelper();
      }),
  );
  return atHelper;
}

describe("useLayoutMutations auto segment fallback", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    segmentPart.mockResolvedValue({ job_id: "cloud-job-1" });
    listPartLines.mockResolvedValue([]);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
    persistLocalSegment.mockResolvedValue({});
    runLocalInference.mockResolvedValue({
      task: "segment",
      output: SEGMENT_OUTPUT,
    });
  });

  it("falls back to the cloud when the local run fails for a non-abort reason", async () => {
    runLocalInference.mockRejectedValueOnce(
      new Error("WEIGHTS_UNAVAILABLE: model weights are not on disk"),
    );
    const { view, setPairingError } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(segmentPart).toHaveBeenCalledTimes(1);
    expect(setPairingError).not.toHaveBeenCalledWith(
      expect.stringContaining("WEIGHTS_UNAVAILABLE"),
    );
  });

  it("does not touch the cloud when the local run succeeds", async () => {
    const { view } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(persistLocalSegment).toHaveBeenCalledTimes(1);
    expect(segmentPart).not.toHaveBeenCalled();
  });

  it("keeps a persisted local segmentation when the reload that follows it fails", async () => {
    // A blip on the cosmetic reload, after the segmentation is already stored.
    listPartLines.mockRejectedValue(new Error("network hiccup"));
    const { view, setPairingError } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(persistLocalSegment).toHaveBeenCalledTimes(1);
    // Segmenting again would replace Segments that are already saved.
    expect(segmentPart).not.toHaveBeenCalled();
    expect(setPairingError).toHaveBeenCalledWith("network hiccup");
  });

  it("does not run in the cloud when the user cancels the local job", async () => {
    const trackLocalTask: TrackLocalTask = async (_meta, run) => {
      const controller = new AbortController();
      controller.abort();
      // Mirrors BackgroundJobsProvider: the UI-owned controller is aborted and
      // the task rejects with the resulting AbortError.
      await run(controller.signal).catch(() => undefined);
      throw new DOMException("Local job cancelled", "AbortError");
    };
    const { view, setPairingError } = setup({ trackLocalTask });

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(segmentPart).not.toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("cancels a superseded run outright instead of racing it in the cloud", async () => {
    const atHelper = helperRunBlockedUntilAbort();
    const { view, setPairingError } = setup();

    await act(async () => {
      // Double-clicking "Auto segment": the second run supersedes the first.
      const superseded = view.result.current.runAutoSegment();
      await atHelper;
      const winner = view.result.current.runAutoSegment();
      await Promise.all([superseded, winner]);
    });

    // Only the run that took over wrote to the page, and nothing was queued in
    // the cloud behind it.
    expect(persistLocalSegment).toHaveBeenCalledTimes(1);
    expect(view.result.current.segmenting).toBe(false);
    expect(segmentPart).not.toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("keeps reporting the page as busy while the run that took over continues", async () => {
    const atFirstHelper = helperRunBlockedUntilAbort();
    const atSecondHelper = helperRunBlockedUntilAbort();
    const { view } = setup();

    let winner: Promise<void>;
    await act(async () => {
      const superseded = view.result.current.runAutoSegment();
      await atFirstHelper;
      winner = view.result.current.runAutoSegment();
      await atSecondHelper;
      // The superseded run unwinds first; the winner is still in the helper.
      await superseded;
    });

    expect(view.result.current.segmenting).toBe(true);

    await act(async () => {
      view.result.current.abortRunToCloud();
      await winner;
    });

    expect(view.result.current.segmenting).toBe(false);
  });

  it("falls back to the cloud when the run is switched to the cloud mid-flight", async () => {
    const atHelper = helperRunBlockedUntilAbort();
    const { view, setPairingError } = setup();

    await act(async () => {
      const running = view.result.current.runAutoSegment();
      await atHelper;
      view.result.current.abortRunToCloud();
      await running;
    });

    expect(segmentPart).toHaveBeenCalledTimes(1);
    expect(persistLocalSegment).not.toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("reports an actionable error instead of using the cloud under local-only routing", async () => {
    runLocalInference.mockRejectedValueOnce(new Error("helper crashed"));
    const { view, setPairingError } = setup({ cloudInferenceEnabled: false });

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(segmentPart).not.toHaveBeenCalled();
    expect(setPairingError).toHaveBeenCalledWith(
      expect.stringContaining("Local only"),
    );
  });
});
