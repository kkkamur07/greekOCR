import type { JobResponse, JobStatus } from "../../api/client";

export type PageEditorJobKind =
  "segmentation" | "transcription-page" | "transcription-segment";

/** Kraken segment on large pages can exceed the default 120s job wait. */
export const SEGMENT_JOB_TIMEOUT_MS = 200_000;

/**
 * A ceiling, not an expectation: the waiter resolves the moment the job turns
 * terminal, and the platform fails a stuck inference job on its own clock
 * (30-minute running timeout, 4-minute no-callback sweep), so a terminal
 * status always arrives before this fires. The old implicit 120s default was
 * shorter than a real cloud page transcription: the client gave up on jobs
 * that then finished, and the new layer only appeared after a page refresh.
 */
export const TRANSCRIBE_JOB_TIMEOUT_MS = 1_860_000;

export function jobStatusLabel(job: JobResponse): string {
  if (job.status === "pending") return "Queued";
  if (job.status === "running") return "Starting…";
  if (job.status === "waiting") return "Processing…";
  if (job.status === "done") return "Complete";
  if (job.status === "failed") return "Failed";
  if (job.status === "cancelled") return "Cancelled";
  return job.status;
}

export function isTerminalJobStatus(status: JobStatus): boolean {
  return status === "done" || status === "failed" || status === "cancelled";
}

export function pageEditorJobKindLabel(kind: PageEditorJobKind): string {
  switch (kind) {
    case "segmentation":
      return "Segmentation";
    case "transcription-page":
      return "Page OCR";
    case "transcription-segment":
      return "Segment OCR";
    default:
      return "Job";
  }
}
