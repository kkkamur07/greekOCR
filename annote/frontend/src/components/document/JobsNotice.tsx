import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from '../ui/toast';
import { api, type JobResponse, type JobStatus } from '../../api/client';
import { ApiError } from '../../api/errors';

const TERMINAL_STATUSES: JobStatus[] = ['done', 'failed'];

type TrackedJob = {
  jobId: string;
  status: JobStatus;
  type?: JobResponse['type'];
  error?: string | null;
};

function shortId(jobId: string): string {
  return jobId.slice(0, 8);
}

export type JobsNoticeProps = {
  enableTestJobs?: boolean;
};

export function JobsNotice({ enableTestJobs = false }: JobsNoticeProps) {
  const [trackedJobs, setTrackedJobs] = useState<TrackedJob[]>([]);
  const [enqueueing, setEnqueueing] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const failedNotifiedRef = useRef<Set<string>>(new Set());

  const trackJob = useCallback((jobId: string) => {
    setTrackedJobs((prev) => {
      if (prev.some((j) => j.jobId === jobId)) return prev;
      return [...prev, { jobId, status: 'pending' }];
    });
    setExpanded(true);
  }, []);

  const activeJobIds = trackedJobs
    .filter((j) => !TERMINAL_STATUSES.includes(j.status))
    .map((j) => j.jobId)
    .join(',');

  useEffect(() => {
    if (!activeJobIds) return;

    let cancelled = false;

    const poll = async () => {
      const ids = activeJobIds.split(',');
      const results = await Promise.all(
        ids.map(async (jobId) => {
          try {
            return await api.getJob(jobId);
          } catch {
            return null;
          }
        }),
      );

      if (cancelled) return;

      setTrackedJobs((prev) => {
        const next = prev.map((row) => ({ ...row }));
        for (const job of results) {
          if (!job) continue;
          const idx = next.findIndex((r) => r.jobId === job.id);
          if (idx < 0) continue;
          next[idx] = {
            jobId: job.id,
            status: job.status,
            type: job.type,
            error: job.error,
          };
          if (
            job.status === 'failed' &&
            job.error &&
            !failedNotifiedRef.current.has(job.id)
          ) {
            failedNotifiedRef.current.add(job.id);
            toast.error(job.error ? `Job failed: ${job.error}` : 'Job failed');
          }
        }
        return next;
      });
    };

    void poll();
    const interval = window.setInterval(() => void poll(), 1500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeJobIds]);

  const handleRunTestJob = async () => {
    setEnqueueing(true);
    try {
      const { job_id } = await api.enqueueTestJob();
      trackJob(job_id);
      toast.success(`Test job enqueued (${shortId(job_id)})`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to enqueue test job';
      toast.error(msg);
    } finally {
      setEnqueueing(false);
    }
  };

  const activeCount = trackedJobs.filter((j) => !TERMINAL_STATUSES.includes(j.status)).length;
  const statusHint =
    activeCount > 0
      ? `${activeCount} running`
      : trackedJobs.length > 0
        ? 'All jobs complete'
        : 'Segment, transcribe, or refine layout';

  return (
    <div>
      <div className="notice-inline" role="status">
        <strong>Jobs</strong>
        <span>{statusHint}</span>
        <span className="spacer" />
        {enableTestJobs && (
          <button
            type="button"
            className="btn btn-outline btn-xs"
            onClick={() => void handleRunTestJob()}
            disabled={enqueueing}
          >
            {enqueueing ? 'Running…' : 'Run job'}
          </button>
        )}
        {trackedJobs.length > 0 && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {expanded ? 'Hide' : 'Details'}
          </button>
        )}
      </div>
      {expanded && trackedJobs.length > 0 && (
        <ul className="part-list" style={{ marginBottom: 20 }} aria-label="Job status">
          {trackedJobs.map((job) => (
            <li key={job.jobId} className="part-row" style={{ listStyle: 'none' }}>
              <div className="part-info">
                <div className="part-num">{shortId(job.jobId)}</div>
                <div className="part-desc">
                  {job.type ?? 'job'} · {job.status}
                  {job.error ? `: ${job.error}` : ''}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
