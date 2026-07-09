import { useEffect, useRef } from 'react';
import { getAccessToken } from '../auth/storage';
import { API_BASE_URL, api, type JobResponse } from '../api/client';
import {
  fetchJobsById,
  JOB_NOTICE_POLL_INTERVAL_MS,
  watchJobViaSse,
} from '../utils/jobPolling';

export {
  JOB_NOTICE_POLL_INTERVAL_MS,
  JOB_WAIT_POLL_INTERVAL_MS,
} from '../utils/jobPolling';

export function useJobPolling(
  jobIds: string[],
  onUpdate: (jobs: JobResponse[]) => void,
  options?: { intervalMs?: number; enabled?: boolean },
): void {
  const intervalMs = options?.intervalMs ?? JOB_NOTICE_POLL_INTERVAL_MS;
  const enabled = options?.enabled ?? jobIds.length > 0;
  const onUpdateRef = useRef(onUpdate);
  onUpdateRef.current = onUpdate;
  const activeKey = jobIds.join(',');

  useEffect(() => {
    if (!enabled || !activeKey) return;

    let cancelled = false;
    const ids = activeKey.split(',');
    const pollFallbackIds = new Set<string>();
    const cleanups: (() => void)[] = [];
    let intervalId: number | undefined;

    const emitUpdates = (jobs: JobResponse[]) => {
      if (cancelled || jobs.length === 0) return;
      onUpdateRef.current(jobs);
    };

    const poll = async () => {
      const pollIds = ids.filter((id) => pollFallbackIds.has(id));
      if (pollIds.length === 0) return;
      const results = await fetchJobsById(api.getJob, pollIds);
      if (cancelled) return;
      const jobs = results.filter((job): job is JobResponse => job !== null);
      emitUpdates(jobs);
    };

    const ensurePolling = () => {
      if (intervalId !== undefined) return;
      intervalId = window.setInterval(() => void poll(), intervalMs);
    };

    const markFallback = (jobId: string) => {
      if (pollFallbackIds.has(jobId)) return;
      pollFallbackIds.add(jobId);
      ensurePolling();
      void poll();
    };

    const sseAvailable =
      typeof fetch !== 'undefined' && typeof ReadableStream !== 'undefined';
    const token = getAccessToken();

    if (!sseAvailable) {
      for (const jobId of ids) {
        pollFallbackIds.add(jobId);
      }
      ensurePolling();
      void poll();
    } else {
      for (const jobId of ids) {
        cleanups.push(
          watchJobViaSse(jobId, {
            eventsUrl: `${API_BASE_URL}/jobs/${jobId}/events`,
            token,
            onUpdate: (job) => emitUpdates([job]),
            onUnavailable: () => markFallback(jobId),
          }),
        );
      }
    }

    return () => {
      cancelled = true;
      for (const cleanup of cleanups) {
        cleanup();
      }
      if (intervalId !== undefined) {
        window.clearInterval(intervalId);
      }
    };
  }, [activeKey, enabled, intervalMs]);
}
