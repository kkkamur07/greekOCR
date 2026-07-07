import { useEffect, useRef } from 'react';
import { api, type JobResponse } from '../api/client';
import { fetchJobsById, JOB_NOTICE_POLL_INTERVAL_MS } from '../utils/jobPolling';

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

    const poll = async () => {
      const ids = activeKey.split(',');
      const results = await fetchJobsById(api.getJob, ids);
      if (cancelled) return;
      const jobs = results.filter((job): job is JobResponse => job !== null);
      onUpdateRef.current(jobs);
    };

    void poll();
    const interval = window.setInterval(() => void poll(), intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeKey, enabled, intervalMs]);
}
