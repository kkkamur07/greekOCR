import { useMemo } from "react";
import { useBackgroundJobs } from "../../../context/BackgroundJobsContext";
import {
  getPageEditorProcessingLabel,
  type PageEditorProcessingKind,
} from "../PageEditorProcessingBanner";

type PageEditorRunStateInput = {
  segmenting: boolean;
  ocrRunning: boolean;
  ocrScope: "segment" | "page" | null;
};

export type PageEditorRunState = {
  /** Which operation the page is busy with, or `null` when it is idle. */
  processingKind: PageEditorProcessingKind;
  /** Sentence for the canvas hint, or `null` when idle. */
  processingLabel: string | null;
  /** Whether the app-wide queue is working on anything at all. */
  backgroundJobsActive: boolean;
};

/**
 * The single answer to "what is this page running right now".
 *
 * `segmenting` / `ocrRunning` stay owned by the hooks that drive the operations:
 * they cover the whole call, including the request that enqueues the job and the
 * reloads that follow it, neither of which the job queue can see. Everything the
 * editor merely *derives* from them is derived here, once, instead of being
 * re-expressed at each use site.
 */
export function usePageEditorRunState({
  segmenting,
  ocrRunning,
  ocrScope,
}: PageEditorRunStateInput): PageEditorRunState {
  const { activeCount } = useBackgroundJobs();

  return useMemo(() => {
    const processingKind: PageEditorProcessingKind = segmenting
      ? "segmentation"
      : ocrRunning
        ? ocrScope === "page"
          ? "transcription-page"
          : "transcription-segment"
        : null;
    return {
      processingKind,
      processingLabel: getPageEditorProcessingLabel(processingKind),
      backgroundJobsActive: activeCount > 0,
    };
  }, [activeCount, ocrRunning, ocrScope, segmenting]);
}
