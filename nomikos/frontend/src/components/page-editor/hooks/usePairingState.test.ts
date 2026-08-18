/**
 * What survives the deletion of the local-first OCR path (#60).
 *
 * The local run, its three abort causes, and the fallback from the helper to
 * the cloud were the loopback transport, and are deleted rather than rewritten.
 * The refusal rule was never about the transport: when no **inference host**
 * has **capacity** the platform answers 409 with a sentence the researcher can
 * act on, and it belongs on the page rather than in the generic error line.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../../api/errors";
import { queryClient, taggedMeta } from "../../../api/queryClient";
import { resourceTags } from "../../../api/resources";
import { platformNoCapacityMessage } from "../../../inference/platformMessages";
import { usePairingState } from "./usePairingState";

const enqueueTranscribePart = vi.fn();
const listPartLines = vi.fn();
const listTranscriptions = vi.fn();
const getPagePairing = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    enqueueTranscribePart: (...args: unknown[]) =>
      enqueueTranscribePart(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    listTranscriptions: (...args: unknown[]) => listTranscriptions(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
  },
}));

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

function setup() {
  const setPairingError = vi.fn();
  const setSubmissionRefusal = vi.fn();
  const trackJobAndWait = vi.fn().mockResolvedValue({
    status: "done",
    result: {
      transcription_id: "transcription-1",
      lines: [{ line_id: "line-1", text: "αβγ", confidence: 0.9 }],
    },
  });

  const view = renderHook(() =>
    usePairingState({
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
      selectedTranscribeModelId: "model-1",
      setSubmissionRefusal,
      trackJobAndWait,
    }),
  );

  return { view, setPairingError, setSubmissionRefusal, trackJobAndWait };
}

/**
 * A cached read of the published page, as `PublicDocumentPage` holds it. The
 * editor never renders it, which is why nothing here used to notice that a
 * transcription had just changed what a reader sees.
 */
const PUBLIC_DOCUMENT_KEY = ["public-document", "project-1", "document-1"];

async function seedPublishedPageRead() {
  await queryClient.fetchQuery({
    queryKey: PUBLIC_DOCUMENT_KEY,
    queryFn: () => Promise.resolve({ name: "before the run" }),
    meta: taggedMeta([resourceTags.publicDocument("project-1", "document-1")]),
  });
}

function publishedPageReadIsStale(): boolean {
  return queryClient.getQueryState(PUBLIC_DOCUMENT_KEY)?.isInvalidated ?? false;
}

describe("usePairingState OCR", () => {
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

  it("explains a refused submission instead of reporting a generic failure", async () => {
    enqueueTranscribePart.mockRejectedValueOnce(
      new ApiError(platformNoCapacityMessage(), 409),
    );
    const { view, setPairingError, setSubmissionRefusal } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(setSubmissionRefusal).toHaveBeenCalledWith(
      platformNoCapacityMessage(),
    );
    expect(setPairingError).not.toHaveBeenCalledWith(
      platformNoCapacityMessage(),
    );
  });

  it("makes the published page stale once a transcription lands", async () => {
    await seedPublishedPageRead();
    const { view } = setup();
    expect(publishedPageReadIsStale()).toBe(false);

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    // The reader's copy of this document now has different text in it.
    expect(publishedPageReadIsStale()).toBe(true);
  });

  it("leaves the published page alone when the submission is refused", async () => {
    enqueueTranscribePart.mockRejectedValueOnce(
      new ApiError(platformNoCapacityMessage(), 409),
    );
    await seedPublishedPageRead();
    const { view } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    // Nothing was written, so nothing is stale.
    expect(publishedPageReadIsStale()).toBe(false);
  });

  it("stays quiet when the researcher cancels the job", async () => {
    enqueueTranscribePart.mockRejectedValueOnce(
      new DOMException("The operation was aborted.", "AbortError"),
    );
    const { view, setPairingError, setSubmissionRefusal } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
    expect(setSubmissionRefusal).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("does not name a host in the message it reports", async () => {
    const { view } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(view.result.current.ocrMessage?.text).not.toMatch(
      /\(local\)|locally|in the cloud/i,
    );
  });

  it("waits with a budget that outlasts a real cloud transcription", async () => {
    // The implicit 120s default was shorter than a cloud page run: the waiter
    // gave up on jobs that then finished, and the new layer only showed after
    // a page refresh.
    const { view, trackJobAndWait } = setup();

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    const options = trackJobAndWait.mock.calls[0]?.[2] as {
      timeoutMs?: number;
    };
    expect(options?.timeoutMs).toBeGreaterThan(120_000);
  });

  it("reloads the layer the job created when the result is unreadable", async () => {
    const { view, trackJobAndWait, setPairingError } = setup();
    trackJobAndWait.mockResolvedValueOnce({
      id: "cloud-job-1",
      status: "done",
      result: null,
    });
    listTranscriptions.mockResolvedValue([
      {
        id: "transcription-2",
        created_by_job_id: "cloud-job-1",
        kind: "model",
      },
    ]);

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    // The layer was committed before the job reported done, so the page must
    // reload it rather than refuse and strand the researcher on stale text.
    expect(listPartLines).toHaveBeenCalled();
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
    expect(view.result.current.ocrMessage?.text).toMatch(/completed/i);
  });
});
