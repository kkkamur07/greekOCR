import {
  executionAnnouncement,
  type JobExecution,
} from "../inference/executionTarget";

type JobExecutionAnnouncementProps = {
  execution: JobExecution | null;
  className?: string;
};

/**
 * Where one job runs, stated on the job itself.
 *
 * This is the entire user interface for **execution target** (ADR 0002), which
 * is why it is a standing line on the job and not a toast: a researcher who
 * looked away must still be able to read where their work went, whether it is
 * queued, running, finished, or failed.
 *
 * `execution` is nullable because a job the browser has only just enqueued -
 * or a purely local task the platform never saw - has nothing to announce yet.
 * Rendering nothing is right there; inventing "cloud" would be a claim.
 */
export function JobExecutionAnnouncement({
  execution,
  className,
}: JobExecutionAnnouncementProps) {
  if (!execution) return null;
  return (
    <p
      className={className ?? "job-execution-announcement"}
      data-execution-target={execution.execution_target}
      data-execution-substituted={
        execution.execution_target_substituted ? "true" : "false"
      }
    >
      {executionAnnouncement(execution)}
    </p>
  );
}
