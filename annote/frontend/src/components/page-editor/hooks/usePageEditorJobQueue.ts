import { useCallback, useEffect, useRef, useState } from 'react';
import { waitForJob, type JobResponse, type JobStatus } from '../../../api/client';
import {
  isTerminalJobStatus,
  jobStatusLabel,
  type PageEditorJobKind,
} from '../jobProgress';

export type TrackedPageEditorJob = {
  id: string;
  label: string;
  kind: PageEditorJobKind;
  status: JobStatus;
  error: string | null;
  progressLabel: string;
  finishedAt: number | null;
};

const COMPLETED_TTL_MS = 10_000;

export function usePageEditorJobQueue() {
  const [jobs, setJobs] = useState<TrackedPageEditorJob[]>([]);
  const [panelExpanded, setPanelExpanded] = useState(true);
  const timersRef = useRef<Map<string, number>>(new Map());

  const scheduleRemoval = useCallback((jobId: string) => {
    const existing = timersRef.current.get(jobId);
    if (existing) window.clearTimeout(existing);
    const timer = window.setTimeout(() => {
      setJobs((current) => current.filter((job) => job.id !== jobId));
      timersRef.current.delete(jobId);
    }, COMPLETED_TTL_MS);
    timersRef.current.set(jobId, timer);
  }, []);

  useEffect(
    () => () => {
      for (const timer of timersRef.current.values()) {
        window.clearTimeout(timer);
      }
      timersRef.current.clear();
    },
    [],
  );

  const trackAndWait = useCallback(
    async (
      jobId: string,
      meta: { label: string; kind: PageEditorJobKind },
    ): Promise<JobResponse> => {
      setJobs((current) => {
        if (current.some((job) => job.id === jobId)) return current;
        return [
          ...current,
          {
            id: jobId,
            label: meta.label,
            kind: meta.kind,
            status: 'pending' as JobStatus,
            error: null,
            progressLabel: 'Queued',
            finishedAt: null,
          },
        ];
      });
      setPanelExpanded(true);

      const patchJob = (latest: JobResponse) => {
        setJobs((current) =>
          current.map((job) =>
            job.id === jobId
              ? {
                  ...job,
                  status: latest.status,
                  error: latest.error,
                  progressLabel: jobStatusLabel(latest),
                  finishedAt: isTerminalJobStatus(latest.status) ? Date.now() : null,
                }
              : job,
          ),
        );
        if (isTerminalJobStatus(latest.status)) {
          scheduleRemoval(jobId);
        }
      };

      try {
        const job = await waitForJob(jobId, { onUpdate: patchJob });
        patchJob(job);
        return job;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Job failed';
        setJobs((current) =>
          current.map((job) =>
            job.id === jobId
              ? {
                  ...job,
                  status: 'failed',
                  error: message,
                  progressLabel: 'Failed',
                  finishedAt: Date.now(),
                }
              : job,
          ),
        );
        scheduleRemoval(jobId);
        throw err;
      }
    },
    [scheduleRemoval],
  );

  const activeCount = jobs.filter((job) => !isTerminalJobStatus(job.status)).length;

  const dismissCompleted = useCallback(() => {
    setJobs((current) => current.filter((job) => !isTerminalJobStatus(job.status)));
  }, []);

  return {
    jobs,
    activeCount,
    panelExpanded,
    setPanelExpanded,
    trackAndWait,
    dismissCompleted,
  };
}
