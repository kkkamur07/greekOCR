import type { JobResponse } from '../api/client';

export const JOB_NOTICE_POLL_INTERVAL_MS = 1500;
export const JOB_WAIT_POLL_INTERVAL_MS = 250;

type GetJob = (jobId: string) => Promise<JobResponse>;

type WaitForJobOptions = {
  timeoutMs?: number;
  intervalMs?: number;
  onUpdate?: (job: JobResponse) => void;
};

type WaitForJobSseOptions = Omit<WaitForJobOptions, 'intervalMs'> & {
  eventsUrl: string;
  token?: string | null;
};

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
  options?: WaitForJobOptions,
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

function parseSseChunk(buffer: string): { events: string[]; remainder: string } {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const parts = normalized.split('\n\n');
  const remainder = parts.pop() ?? '';
  const events = parts
    .map((part) =>
      part
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n'),
    )
    .filter(Boolean);
  return { events, remainder };
}

export type WatchJobSseOptions = {
  eventsUrl: string;
  token?: string | null;
  onUpdate: (job: JobResponse) => void;
  onUnavailable?: () => void;
};

export function watchJobViaSse(_jobId: string, options: WatchJobSseOptions): () => void {
  if (typeof fetch === 'undefined' || typeof ReadableStream === 'undefined') {
    options.onUnavailable?.();
    return () => {};
  }

  const controller = new AbortController();
  let closed = false;

  const cleanup = () => {
    if (closed) return;
    closed = true;
    controller.abort();
  };

  void (async () => {
    try {
      const headers = new Headers({ Accept: 'text/event-stream' });
      if (options.token) {
        headers.set('Authorization', `Bearer ${options.token}`);
      }

      const response = await fetch(options.eventsUrl, {
        headers,
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        if (!closed) options.onUnavailable?.();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let lastJob: JobResponse | null = null;

      while (!closed) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseChunk(buffer);
        buffer = parsed.remainder;

        for (const event of parsed.events) {
          const job = JSON.parse(event) as JobResponse;
          if (!lastJob || lastJob.status !== job.status || lastJob.updated_at !== job.updated_at) {
            options.onUpdate(job);
          }
          lastJob = job;
          if (job.status === 'done' || job.status === 'failed') {
            cleanup();
            return;
          }
        }
      }

      if (!closed && (!lastJob || (lastJob.status !== 'done' && lastJob.status !== 'failed'))) {
        options.onUnavailable?.();
      }
    } catch (err) {
      if (!closed && !(err instanceof DOMException && err.name === 'AbortError')) {
        options.onUnavailable?.();
      }
    }
  })();

  return cleanup;
}

export async function waitForJobViaSse(
  jobId: string,
  options: WaitForJobSseOptions,
): Promise<JobResponse> {
  if (typeof fetch === 'undefined' || typeof ReadableStream === 'undefined') {
    throw new Error('SSE streaming is not available in this browser.');
  }

  const timeoutMs = options.timeoutMs ?? 120_000;
  const controller = new AbortController();
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const headers = new Headers({ Accept: 'text/event-stream' });
    if (options.token) {
      headers.set('Authorization', `Bearer ${options.token}`);
    }

    const response = await fetch(options.eventsUrl, {
      headers,
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error('SSE job stream could not be opened.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let lastJob: JobResponse | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSseChunk(buffer);
      buffer = parsed.remainder;

      for (const event of parsed.events) {
        const job = JSON.parse(event) as JobResponse;
        if (!lastJob || lastJob.status !== job.status || lastJob.updated_at !== job.updated_at) {
          options.onUpdate?.(job);
        }
        lastJob = job;
        if (job.status === 'done') return job;
        if (job.status === 'failed') {
          throw new Error(job.error ?? 'Job failed.');
        }
      }
    }
  } catch (err) {
    if (timedOut) {
      throw new Error('Job timed out.');
    }
    throw err;
  } finally {
    window.clearTimeout(timeout);
  }

  throw new Error(`SSE job stream for ${jobId} closed before the job completed.`);
}
