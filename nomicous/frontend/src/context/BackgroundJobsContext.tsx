import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  waitForJob,
  type JobResponse,
  type JobStatus,
} from "../api/client";
import { toast } from "../components/ui/toast";
import { useJobPolling } from "../hooks/useJobPolling";
import {
  isTerminalJobStatus,
  jobStatusLabel,
  type PageEditorJobKind,
} from "../components/page-editor/jobProgress";
import { jobExecution, type JobExecution } from "../inference/executionTarget";

export type TrackedBackgroundJob = {
  id: string;
  label: string;
  kind: PageEditorJobKind;
  status: JobStatus;
  error: string | null;
  progressLabel: string;
  finishedAt: number | null;
  /**
   * Which **inference host** the platform fixed for this job, and which one the
   * account asked for. `null` until the platform has answered.
   */
  execution: JobExecution | null;
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
  cancelJob: (jobId: string) => Promise<void>;
  dismissCompleted: () => void;
};

const COMPLETED_TTL_MS = 10_000;

const BackgroundJobsContext = createContext<BackgroundJobsContextValue | null>(
  null,
);

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
    // Re-read on every poll rather than only at submission: the host is fixed,
    // but the *status* is part of the announcement ("Failed on your computer").
    execution: jobExecution(latest),
  };
}

export function BackgroundJobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<TrackedBackgroundJob[]>([]);
  const [panelExpanded, setPanelExpanded] = useState(false);
  const timersRef = useRef<Map<string, number>>(new Map());

  const scheduleRemoval = useCallback((jobId: string, status?: JobStatus) => {
    // Keep cancelled jobs visible until the user clears them - they are part
    // of recent history, unlike done/failed which auto-dismiss after a TTL.
    if (status === "cancelled") return;
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
        current.map((job) =>
          job.id === jobId ? patchTrackedJob(job, latest) : job,
        ),
      );
      if (isTerminalJobStatus(latest.status)) {
        scheduleRemoval(jobId, latest.status);
      }
    },
    [scheduleRemoval],
  );

  const activeJobIds = useMemo(
    () =>
      jobs
        .filter((job) => !isTerminalJobStatus(job.status))
        .map((job) => job.id),
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
          scheduleRemoval(update.id, update.status);
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
            status: "pending" as JobStatus,
            error: null,
            progressLabel: "Waiting for allocation…",
            finishedAt: null,
            // Filled by the first update from the platform. Enqueueing returns
            // only an id, and guessing a host would be a claim, not a report.
            execution: null,
          },
        ];
      });
      setPanelExpanded(false);

      try {
        const job = await waitForJob(jobId, {
          timeoutMs: options?.timeoutMs,
          onUpdate: (latest) => applyJobUpdate(jobId, latest),
        });
        applyJobUpdate(jobId, job);
        return job;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Job failed";
        setJobs((current) =>
          current.map((job) =>
            job.id === jobId
              ? {
                  ...job,
                  status: "failed",
                  error: message,
                  progressLabel: "Failed",
                  finishedAt: Date.now(),
                  // A failed job says which host it failed on, so the host it
                  // was given has to travel with the new status.
                  execution: job.execution
                    ? { ...job.execution, status: "failed" }
                    : null,
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
    setJobs((current) =>
      current.filter((job) => !isTerminalJobStatus(job.status)),
    );
  }, []);

  const cancelJob = useCallback(
    async (jobId: string) => {
      try {
        const latest = await api.cancelJob(jobId);
        applyJobUpdate(jobId, latest);
        toast.success("Job cancelled");
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Could not cancel that job",
        );
        throw err;
      }
    },
    [applyJobUpdate],
  );

  const activeCount = jobs.filter(
    (job) => !isTerminalJobStatus(job.status),
  ).length;

  const value = useMemo(
    () => ({
      jobs,
      activeCount,
      panelExpanded,
      setPanelExpanded,
      trackAndWait,
      cancelJob,
      dismissCompleted,
    }),
    [
      jobs,
      activeCount,
      panelExpanded,
      trackAndWait,
      cancelJob,
      dismissCompleted,
    ],
  );

  return (
    <BackgroundJobsContext.Provider value={value}>
      {children}
    </BackgroundJobsContext.Provider>
  );
}

export function useBackgroundJobs(): BackgroundJobsContextValue {
  const context = useContext(BackgroundJobsContext);
  if (!context) {
    throw new Error(
      "useBackgroundJobs must be used within BackgroundJobsProvider",
    );
  }
  return context;
}
