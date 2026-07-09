import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { waitForJob, type JobResponse, type JobStatus } from '../api/client';
import { useJobPolling } from '../hooks/useJobPolling';
import {
  isTerminalJobStatus,
  jobStatusLabel,
  type PageEditorJobKind,
} from '../components/page-editor/jobProgress';

export type TrackedBackgroundJob = {
  id: string;
  label: string;
  kind: PageEditorJobKind;
  status: JobStatus;
  error: string | null;
  progressLabel: string;
  finishedAt: number | null;
};

type BackgroundJobsContextValue = {
  jobs: TrackedBackgroundJob[];
  activeCount: number;
  panelExpanded: boolean;
  setPanelExpanded: (expanded: boolean) => void;
  trackAndWait: (
    jobId: string,
    meta: { label: string; kind: PageEditorJobKind },
    options?: { timeoutMs?: number },
  ) => Promise<JobResponse>;
  dismissCompleted: () => void;
};

const COMPLETED_TTL_MS = 10_000;

const BackgroundJobsContext = createContext<BackgroundJobsContextValue | null>(null);

function patchTrackedJob(
  job: TrackedBackgroundJob,
  latest: JobResponse,
): TrackedBackgroundJob {
  return {
    ...job,
    status: latest.status,
    error: latest.error,
    progressLabel: jobStatusLabel(latest),
    finishedAt: isTerminalJobStatus(latest.status) ? Date.now() : null,
  };
}

export function BackgroundJobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<TrackedBackgroundJob[]>([]);
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

  const applyJobUpdate = useCallback(
    (jobId: string, latest: JobResponse) => {
      setJobs((current) =>
        current.map((job) => (job.id === jobId ? patchTrackedJob(job, latest) : job)),
      );
      if (isTerminalJobStatus(latest.status)) {
        scheduleRemoval(jobId);
      }
    },
    [scheduleRemoval],
  );

  const activeJobIds = useMemo(
    () => jobs.filter((job) => !isTerminalJobStatus(job.status)).map((job) => job.id),
    [jobs],
  );

  useJobPolling(activeJobIds, (updates) => {
    setJobs((current) => {
      const next = current.map((job) => ({ ...job }));
      for (const update of updates) {
        const index = next.findIndex((job) => job.id === update.id);
        if (index < 0) continue;
        next[index] = patchTrackedJob(next[index], update);
        if (isTerminalJobStatus(update.status)) {
          scheduleRemoval(update.id);
        }
      }
      return next;
    });
  });

  const trackAndWait = useCallback(
    async (
      jobId: string,
      meta: { label: string; kind: PageEditorJobKind },
      options?: { timeoutMs?: number },
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

      try {
        const job = await waitForJob(jobId, {
          timeoutMs: options?.timeoutMs,
          onUpdate: (latest) => applyJobUpdate(jobId, latest),
        });
        applyJobUpdate(jobId, job);
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
    [applyJobUpdate, scheduleRemoval],
  );

  const dismissCompleted = useCallback(() => {
    setJobs((current) => current.filter((job) => !isTerminalJobStatus(job.status)));
  }, []);

  const activeCount = jobs.filter((job) => !isTerminalJobStatus(job.status)).length;

  const value = useMemo(
    () => ({
      jobs,
      activeCount,
      panelExpanded,
      setPanelExpanded,
      trackAndWait,
      dismissCompleted,
    }),
    [jobs, activeCount, panelExpanded, trackAndWait, dismissCompleted],
  );

  return (
    <BackgroundJobsContext.Provider value={value}>{children}</BackgroundJobsContext.Provider>
  );
}

export function useBackgroundJobs(): BackgroundJobsContextValue {
  const context = useContext(BackgroundJobsContext);
  if (!context) {
    throw new Error('useBackgroundJobs must be used within BackgroundJobsProvider');
  }
  return context;
}
