import type { JobResponse, JobStatus } from "../../api/client";

export type PageEditorJobKind =
  "segmentation" | "transcription-page" | "transcription-segment";

/**
 * How long the editor listens for an inference job's terminal status: one
 * ceiling for segmentation and transcription alike.
 *
 * A ceiling, not an expectation. The platform reaches a verdict on its own
 * clock: a submission with no host is refused up front (409), a dispatched
 * job whose callback never comes is failed by the 4-minute waiting sweep, and
 * an abandoned agent claim is re-pended and retried until the platform fails
 * the page. A truthful terminal status always arrives well before this fires.
 *
 * Bug this avoids: the previous timeouts (120s transcription, 200s
 * segmentation) were shorter than the 240s server-side waiting sweep, so the
 * client gave up on jobs that later finished, and the result only appeared
 * after a manual refresh. The client deadline must never undercut the
 * server's.
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
