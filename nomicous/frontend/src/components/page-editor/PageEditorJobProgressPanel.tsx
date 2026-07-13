import type { TrackedBackgroundJob } from "../../context/BackgroundJobsContext";
import { isTerminalJobStatus, pageEditorJobKindLabel } from "./jobProgress";

type PageEditorJobProgressPanelProps = {
  jobs: TrackedBackgroundJob[];
  activeCount: number;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  onDismissCompleted: () => void;
  onCancelJob: (jobId: string) => void;
};

function statusClass(status: TrackedBackgroundJob["status"]): string {
  if (status === "done") return "pe-job-item--done";
  if (status === "failed") return "pe-job-item--failed";
  if (status === "cancelled") return "pe-job-item--failed";
  return "pe-job-item--active";
}

export function PageEditorJobProgressPanel({
  jobs,
  activeCount,
  expanded,
  onExpandedChange,
  onDismissCompleted,
  onCancelJob,
}: PageEditorJobProgressPanelProps) {
  if (jobs.length === 0) return null;

  const completedCount = jobs.filter((job) =>
    isTerminalJobStatus(job.status),
  ).length;

  if (!expanded) {
    const statusLabel =
      activeCount > 0
        ? `${activeCount} background job${activeCount === 1 ? "" : "s"} running`
        : `${completedCount} background job${completedCount === 1 ? "" : "s"} finished`;
    return (
      <button
        type="button"
        className="pe-job-panel pe-job-panel--collapsed"
        aria-label={statusLabel}
        onClick={() => onExpandedChange(true)}
      >
        <span className="pe-job-panel__pulse" aria-hidden="true" />
        {activeCount > 0
          ? `${activeCount} job${activeCount === 1 ? "" : "s"} running`
          : `${completedCount} job${completedCount === 1 ? "" : "s"} finished`}
      </button>
    );
  }

  return (
    <section
      className="pe-job-panel"
      role="dialog"
      aria-label="Background jobs"
      aria-live="polite"
    >
      <header className="pe-job-panel__head">
        <div>
          <h2 className="pe-job-panel__title">Background jobs</h2>
          <p className="pe-job-panel__subtitle">
            {activeCount > 0
              ? `${activeCount} in progress. You can keep editing other segments.`
              : "Recent jobs"}
          </p>
        </div>
        <div className="pe-job-panel__head-actions">
          {completedCount > 0 && activeCount === 0 && (
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={onDismissCompleted}
            >
              Clear
            </button>
          )}
          <button
            type="button"
            className="pe-tb-btn"
            aria-label="Minimize jobs panel"
            title="Minimize"
            onClick={() => onExpandedChange(false)}
          >
            −
          </button>
        </div>
      </header>
      <ul className="pe-job-panel__list">
        {jobs.map((job) => (
          <li key={job.id} className={`pe-job-item ${statusClass(job.status)}`}>
            {!isTerminalJobStatus(job.status) && (
              <span className="pe-job-item__spinner" aria-hidden="true" />
            )}
            <div className="pe-job-item__body">
              <div className="pe-job-item__row">
                <span className="pe-job-item__kind">
                  {pageEditorJobKindLabel(job.kind)}
                </span>
                <span className="pe-job-item__status">{job.progressLabel}</span>
              </div>
              <strong className="pe-job-item__label">{job.label}</strong>
              {job.error && <p className="pe-job-item__error">{job.error}</p>}
              {!isTerminalJobStatus(job.status) && (
                <button
                  type="button"
                  className="btn btn-ghost btn-xs"
                  onClick={() => onCancelJob(job.id)}
                >
                  Cancel
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
