/**
 * What survives the deletion of the local-first write path (#60).
 *
 * Most of this file used to be about the loopback transport: a local run, its
 * three abort causes, and the fallback from the helper to the cloud. There is
 * one path now, so those cases are gone rather than rewritten - a cloud-only
 * "fallback" test would be asserting a decision nothing makes any more.
 *
 * What is left is the pair of rules that were never about the transport: a
 * refused submission is an explanation the researcher can act on, and a failure
 * of the reload *after* a stored segmentation must not be mistaken for a
 * failure of the segmentation.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../../api/errors";
import { platformNoCapacityMessage } from "../../../inference/platformMessages";
import { useLayoutMutations } from "./useLayoutMutations";

const segmentPart = vi.fn();
const listPartLines = vi.fn();
const getPartLayout = vi.fn();
const getPagePairing = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    segmentPart: (...args: unknown[]) => segmentPart(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
  },
}));

function setup() {
  const setPairingError = vi.fn();
  const setSubmissionRefusal = vi.fn();
  const trackJobAndWait = vi.fn().mockResolvedValue({ status: "done" });

  const view = renderHook(() =>
    useLayoutMutations({
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
      setSubmissionRefusal,
      trackJobAndWait,
    }),
  );

  return { view, setPairingError, setSubmissionRefusal, trackJobAndWait };
}

describe("useLayoutMutations auto segment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    segmentPart.mockResolvedValue({ job_id: "cloud-job-1" });
    listPartLines.mockResolvedValue([]);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
  });

  it("submits one job and waits for it", async () => {
    const { view, trackJobAndWait } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(segmentPart).toHaveBeenCalledTimes(1);
    expect(trackJobAndWait).toHaveBeenCalledTimes(1);
  });

  it("keeps a finished segmentation when the reload that follows it fails", async () => {
    // A blip on the cosmetic reload, after the segmentation is already stored.
    listPartLines.mockRejectedValue(new Error("network hiccup"));
    const { view, setPairingError } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // Segmenting again would replace Segments that are already saved.
    expect(segmentPart).toHaveBeenCalledTimes(1);
    expect(setPairingError).toHaveBeenCalledWith("network hiccup");
  });

  it("does not name a host in the message it reports", async () => {
    const { view } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // The job announces its **execution target**; a second sentence here, from
    // a second source, is how the two come to disagree.
    expect(view.result.current.segmentMessage).not.toMatch(
      /locally|in the cloud|on your computer/i,
    );
  });

  it("explains a refused submission instead of reporting a generic failure", async () => {
    segmentPart.mockRejectedValueOnce(
      new ApiError(platformNoCapacityMessage(), 409),
    );
    const { view, setPairingError, setSubmissionRefusal } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(setSubmissionRefusal).toHaveBeenCalledWith(
      platformNoCapacityMessage(),
    );
    expect(setPairingError).not.toHaveBeenCalledWith(
      platformNoCapacityMessage(),
    );
  });

  it("stays quiet when the researcher cancels the job", async () => {
    segmentPart.mockRejectedValueOnce(
      new DOMException("The operation was aborted.", "AbortError"),
    );
    const { view, setPairingError, setSubmissionRefusal } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // The jobs panel already reported it.
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
    expect(setSubmissionRefusal).not.toHaveBeenCalledWith(expect.any(String));
  });
});
