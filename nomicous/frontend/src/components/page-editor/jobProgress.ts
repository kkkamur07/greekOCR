import type { JobResponse, JobStatus } from "../../api/client";

export type PageEditorJobKind =
  "segmentation" | "transcription-page" | "transcription-segment";

/**
 * How long the editor listens for an inference job's terminal status - one
 * ceiling for segmentation and transcription alike.
 *
 * A ceiling, not an expectation: the waiter resolves the moment the job turns
 * terminal, and the platform reaches a verdict on its own clock - a submission
 * with no host is refused up front (409), a dispatched job whose callback
 * never comes is failed by the 4-minute waiting sweep, and an abandoned agent
 * claim is re-pended and retried until the platform itself fails the page. So
 * a truthful terminal status always arrives long before this fires, capacity
 * crunch included.
 *
 * The budgets this replaces were shorter than the platform's own verdicts: an
 * implicit 120s default for transcription and 200s for segmentation - under
 * the 240s waiting sweep, let alone a real cloud page run. The client gave up
 * on jobs that then finished, and the result only appeared after a manual
 * page refresh. A client deadline may never undercut the server's, because
 * every second between them turns a truthful outcome into an invented one.
 */
export const INFERENCE_JOB_WAIT_CEILING_MS = 1_860_000;

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
