import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { api, type DocumentResponse, type JobResponse } from "../../api/client";
import { ApiError } from "../../api/errors";
import { useJobPolling } from "../../hooks/useJobPolling";
import {
  isTerminalJobStatus,
  jobStatusLabel,
} from "../page-editor/jobProgress";

type ProjectJobsPanelProps = {
  projectId: string;
  documents: DocumentResponse[];
};

const VISIBLE_JOB_LIMIT = 8;

function formatJobType(type: JobResponse["type"]): string {
  switch (type) {
    case "segment":
      return "Segmentation";
    case "transcribe":
      return "Transcription";
    case "binarize":
      return "Binarize";
    case "pipeline":
      return "Pipeline";
    default:
      return type;
  }
}

function formatWhen(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatExecution(execution: JobResponse["execution"]): string {
  if (execution === "local") return "Local";
  return "Cloud";
}

function executionClass(execution: JobResponse["execution"]): string {
  if (execution === "local") return "project-jobs-panel__host--local";
  return "project-jobs-panel__host--cloud";
}

function statusClass(status: JobResponse["status"]): string {
  if (status === "failed") return "project-jobs-panel__status--failed";
  if (status === "done") return "project-jobs-panel__status--done";
  if (status === "pending" || status === "waiting")
    return "project-jobs-panel__status--pending";
  return "project-jobs-panel__status--active";
}

export function ProjectJobsPanel({
  projectId,
  documents,
}: ProjectJobsPanelProps) {
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const loadMoreControllerRef = useRef<AbortController | null>(null);

  const documentNames = useMemo(
    () => new Map(documents.map((document) => [document.id, document.name])),
    [documents],
  );

  const load = useCallback(
    async (cursor: string | null, signal: AbortSignal) => {
      const isFirstPage = cursor === null;
      if (isFirstPage) setLoading(true);
      else setLoadingMore(true);
      setError(null);
      try {
        const page = await api.listProjectJobsPage(projectId, {
          cursor,
          limit: VISIBLE_JOB_LIMIT,
          signal,
        });
        if (signal.aborted) return;
        setJobs((current) => {
          const next = isFirstPage ? page.items : [...current, ...page.items];
          return Array.from(new Map(next.map((job) => [job.id, job])).values());
        });
        setNextCursor(page.next_cursor);
      } catch (err) {
        if (signal.aborted) return;
        const message =
          err instanceof ApiError ? err.message : "Failed to load jobs";
        setError(message);
        if (isFirstPage) setJobs([]);
      } finally {
        if (!signal.aborted) {
          if (isFirstPage) setLoading(false);
          else setLoadingMore(false);
        }
      }
    },
    [projectId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(null, controller.signal);
    return () => {
      controller.abort();
      loadMoreControllerRef.current?.abort();
    };
  }, [load]);

  const activeJobIds = jobs
    .filter((job) => !isTerminalJobStatus(job.status))
    .map((job) => job.id);

  useJobPolling(activeJobIds, (updates) => {
    setJobs((current) => {
      const next = [...current];
      for (const update of updates) {
        const index = next.findIndex((job) => job.id === update.id);
        if (index >= 0) {
          next[index] = update;
        }
      }
      return next.sort(
        (left, right) =>
          new Date(right.created_at).getTime() -
          new Date(left.created_at).getTime(),
      );
    });
  });

  const activeCount = jobs.filter(
    (job) => !isTerminalJobStatus(job.status),
  ).length;

  useEffect(() => {
    if (activeCount > 0) {
      setExpanded(true);
    }
  }, [activeCount]);

  const summary = loading
    ? "Loading…"
    : activeCount > 0
      ? `${activeCount} running`
      : jobs.length > 0
        ? `${jobs.length} recent`
        : "No jobs yet";

  const loadMore = () => {
    if (!nextCursor || loadingMore) return;
    const controller = new AbortController();
    loadMoreControllerRef.current?.abort();
    loadMoreControllerRef.current = controller;
    void load(nextCursor, controller.signal).finally(() => {
      if (loadMoreControllerRef.current === controller) {
        loadMoreControllerRef.current = null;
      }
    });
  };

  if (!loading && !error && jobs.length === 0) {
    return (
      <section
        className="project-jobs-panel project-jobs-panel--empty"
        aria-labelledby="project-jobs-heading"
      >
        <h2 className="project-jobs-panel__heading" id="project-jobs-heading">
          Jobs
        </h2>
        <p className="project-jobs-panel__hint">
          No background jobs yet. Segment or transcribe from a document page.
        </p>
      </section>
    );
  }

  return (
    <section
      className="project-jobs-panel"
      aria-labelledby="project-jobs-heading"
    >
      <div className="project-jobs-panel__bar">
        <button
          type="button"
          className="project-jobs-panel__toggle"
          aria-expanded={expanded}
          aria-controls="project-jobs-panel-body"
          onClick={() => setExpanded((value) => !value)}
        >
          <span className="project-jobs-panel__toggle-main">
            <span
              className="project-jobs-panel__heading"
              id="project-jobs-heading"
            >
              Jobs
            </span>
            <span className="project-jobs-panel__summary">{summary}</span>
          </span>
          <span className="project-jobs-panel__chevron" aria-hidden="true">
            {expanded ? "▴" : "▾"}
          </span>
        </button>
      </div>

      {error && (
        <div className="notice-banner" role="alert">
          <strong>Jobs unavailable</strong>
          {error}
        </div>
      )}

      {expanded && !loading && !error && jobs.length > 0 && (
        <div className="project-jobs-panel__body" id="project-jobs-panel-body">
          <div className="data-panel">
            <table className="data-list" aria-label="Project jobs">
              <thead>
                <tr>
                  <th scope="col">Job</th>
                  <th scope="col">Host</th>
                  <th scope="col">Document</th>
                  <th scope="col">Status</th>
                  <th scope="col">Started</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => {
                  const documentName = job.document_id
                    ? (documentNames.get(job.document_id) ??
                      job.document_id.slice(0, 8))
                    : "-";
                  return (
                    <tr key={job.id}>
                      <td>
                        <span className="row-title">
                          {formatJobType(job.type)}
                        </span>
                        <span className="row-sub">{job.id.slice(0, 8)}</span>
                      </td>
                      <td>
                        <span
                          className={`project-jobs-panel__host ${executionClass(job.execution)}`}
                        >
                          {formatExecution(job.execution)}
                        </span>
                      </td>
                      <td className="col-muted">
                        {job.document_id ? (
                          <Link
                            href={`/projects/${projectId}/documents/${job.document_id}`}
                          >
                            {documentName}
                          </Link>
                        ) : (
                          documentName
                        )}
                      </td>
                      <td>
                        <span
                          className={`project-jobs-panel__status ${statusClass(job.status)}`}
                        >
                          {jobStatusLabel(job)}
                        </span>
                        {job.error && (
                          <span className="row-sub" title={job.error}>
                            {job.error}
                          </span>
                        )}
                      </td>
                      <td className="col-muted">
                        {formatWhen(job.created_at)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {nextCursor && (
            <div className="project-jobs-panel__more">
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={loadMore}
                disabled={loadingMore}
              >
                {loadingMore ? "Loading jobs…" : "Load more jobs"}
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
