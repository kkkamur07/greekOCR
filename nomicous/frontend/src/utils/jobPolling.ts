import type { JobResponse } from '../api/client';

export const JOB_NOTICE_POLL_INTERVAL_MS = 1500;
export const JOB_WAIT_POLL_INTERVAL_MS = 250;

type GetJob = (jobId: string) => Promise<JobResponse>;

export async function fetchJobsById(
  getJob: GetJob,
  jobIds: string[],
): Promise<(JobResponse | null)[]> {
  return Promise.all(
    jobIds.map(async (jobId) => {
      try {
        return await getJob(jobId);
      } catch {
        return null;
      }
    }),
  );
}

export async function pollJobUntilTerminal(
  getJob: GetJob,
  jobId: string,
  options?: {
    timeoutMs?: number;
    intervalMs?: number;
    onUpdate?: (job: JobResponse) => void;
  },
): Promise<JobResponse> {
  const timeoutMs = options?.timeoutMs ?? 120_000;
  const intervalMs = options?.intervalMs ?? JOB_WAIT_POLL_INTERVAL_MS;
  const deadline = Date.now() + timeoutMs;
  let lastJob: JobResponse | null = null;

  while (Date.now() < deadline) {
    const job = await getJob(jobId);
    if (!lastJob || lastJob.status !== job.status || lastJob.updated_at !== job.updated_at) {
      options?.onUpdate?.(job);
    }
    lastJob = job;
    if (job.status === 'done') return job;
    if (job.status === 'failed') {
      throw new Error(job.error ?? 'Job failed.');
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error('Job timed out.');
}
