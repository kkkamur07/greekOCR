import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useLocalInferenceRuns } from "../../../inference";
import { usePairingState } from "./usePairingState";

const enqueueTranscribePart = vi.fn();
const persistLocalTranscribe = vi.fn();
const listPartLines = vi.fn();
const listTranscriptions = vi.fn();
const getPagePairing = vi.fn();
const runLocalInference = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    enqueueTranscribePart: (...args: unknown[]) =>
      enqueueTranscribePart(...args),
    persistLocalTranscribe: (...args: unknown[]) =>
      persistLocalTranscribe(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    listTranscriptions: (...args: unknown[]) => listTranscriptions(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
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

const LINE = {
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

const MODEL = {
  id: "model-1",
  name: "Greek Calamari",
  task: "transcribe",
  artifact_ref: "registry://greek-calamari-v1?tag=stable",
};

type TrackLocalTask = <T>(
  meta: { label: string; kind: string },
  run: (signal: AbortSignal) => Promise<T>,
) => Promise<T>;

function setup(options?: {
  cloudInferenceEnabled?: boolean;
  trackLocalTask?: TrackLocalTask;
}) {
  const setPairingError = vi.fn();
  const trackJobAndWait = vi.fn().mockResolvedValue({
    status: "done",
    result: {
      transcription_id: "transcription-1",
      lines: [{ line_id: "line-1", text: "αβγ", confidence: 0.9 }],
    },
  });

  // The real run registry, so the three abort causes are told apart exactly the
  // way the page tells them apart.
  const view = renderHook(() => {
    const { localInference, abortRunToCloud } = useLocalInferenceRuns(
      () => true,
    );
    const pairing = usePairingState({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      lines: [LINE],
      setLines: vi.fn(),
      transcriptionLayers: [],
      setTranscriptionLayers: vi.fn(),
      selectedTranscriptionLayerId: null,
      setSelectedTranscriptionLayerId: vi.fn(),
      groundTruthTranscriptionId: null,
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError,
      selectedTranscribeModelId: MODEL.id,
      transcribeModels: [MODEL],
      partImageUrl: "http://localhost:8000/media/parts/part-1",
      shouldUseLocalPath: () => true,
      cloudInferenceEnabled: options?.cloudInferenceEnabled ?? true,
      localInference,
      trackJobAndWait,
      trackLocalTask:
        options?.trackLocalTask ??
        ((_meta, run) => run(new AbortController().signal)),
    });
    return { ...pairing, abortRunToCloud };
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

describe("usePairingState OCR fallback", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    enqueueTranscribePart.mockResolvedValue({ job_id: "cloud-job-1" });
    listPartLines.mockResolvedValue([LINE]);
    listTranscriptions.mockResolvedValue([]);
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
  });

  it("falls back to the cloud when the local run fails for a non-abort reason", async () => {
    runLocalInference.mockRejectedValue(new Error("503 WEIGHTS_UNAVAILABLE"));
    const { view } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(enqueueTranscribePart).toHaveBeenCalledTimes(1);
  });

  it("cancels a superseded run outright instead of racing it in the cloud", async () => {
    const atHelper = helperRunBlockedUntilAbort();
    runLocalInference.mockResolvedValue({
      task: "transcribe",
      output: {
        lines: [
          {
            line_id: "line-1",
            line_index: 0,
            output: { text: "αβγ", confidence: 0.9, character_confidences: [] },
          },
        ],
      },
    });
    persistLocalTranscribe.mockResolvedValue({
      transcription_id: "transcription-1",
      lines: [{ line_id: "line-1", text: "αβγ", confidence: 0.9 }],
    });
    const { view, setPairingError } = setup();

    await act(async () => {
      const superseded = view.result.current.runPageOcr();
      await atHelper;
      const winner = view.result.current.runPageOcr();
      await Promise.all([superseded, winner]);
    });

    expect(persistLocalTranscribe).toHaveBeenCalledTimes(1);
    expect(enqueueTranscribePart).not.toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("falls back to the cloud when the run is switched to the cloud mid-flight", async () => {
    const atHelper = helperRunBlockedUntilAbort();
    const { view, setPairingError } = setup();

    await act(async () => {
      const running = view.result.current.runPageOcr();
      await atHelper;
      view.result.current.abortRunToCloud();
      await running;
    });

    expect(enqueueTranscribePart).toHaveBeenCalledTimes(1);
    expect(persistLocalTranscribe).not.toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("does not run in the cloud when the user cancels the local job", async () => {
    const { view, setPairingError } = setup({
      trackLocalTask: async (_meta, run) => {
        const controller = new AbortController();
        controller.abort();
        // Mirrors BackgroundJobsProvider: the UI-owned controller is aborted and
        // the task rejects with the resulting AbortError.
        await run(controller.signal).catch(() => undefined);
        throw new DOMException("Local job cancelled", "AbortError");
      },
    });

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(enqueueTranscribePart).not.toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("keeps a persisted local result when the reload that follows it fails", async () => {
    runLocalInference.mockResolvedValue({
      task: "transcribe",
      output: {
        lines: [
          {
            line_id: "line-1",
            line_index: 0,
            output: { text: "αβγ", confidence: 0.9, character_confidences: [] },
          },
        ],
      },
    });
    persistLocalTranscribe.mockResolvedValue({
      transcription_id: "transcription-1",
      lines: [{ line_id: "line-1", text: "αβγ", confidence: 0.9 }],
    });
    // A blip on the cosmetic reload, after the transcription is already stored.
    listPartLines.mockRejectedValue(new Error("network hiccup"));
    listTranscriptions.mockRejectedValue(new Error("network hiccup"));
    const { view, setPairingError } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(persistLocalTranscribe).toHaveBeenCalledTimes(1);
    // The work succeeded and was billed once; a stale view must never buy a
    // second cloud transcription of the same page.
    expect(enqueueTranscribePart).not.toHaveBeenCalled();
    expect(setPairingError).toHaveBeenCalledWith("network hiccup");
  });

  it("reports an actionable error instead of using the cloud under local-only routing", async () => {
    runLocalInference.mockRejectedValue(new Error("helper crashed"));
    const { view, setPairingError } = setup({ cloudInferenceEnabled: false });

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(enqueueTranscribePart).not.toHaveBeenCalled();
    expect(setPairingError).toHaveBeenCalledWith(
      expect.stringContaining("Local only"),
    );
  });
});
