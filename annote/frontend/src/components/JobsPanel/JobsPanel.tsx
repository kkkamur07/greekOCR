import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, Card, List, Space, Tag, Typography } from 'antd';
import { ThunderboltOutlined } from '@ant-design/icons';
import { api, type JobResponse, type JobStatus } from '../../api/client';
import { ApiError } from '../../api/errors';
import { toast } from '../ui/toast';

const TERMINAL_STATUSES: JobStatus[] = ['done', 'failed'];

const STATUS_COLORS: Record<JobStatus, string> = {
  pending: 'default',
  running: 'processing',
  done: 'success',
  failed: 'error',
};

type TrackedJob = {
  jobId: string;
  status: JobStatus;
  type?: JobResponse['type'];
  error?: string | null;
};

function shortId(jobId: string): string {
  return jobId.slice(0, 8);
}

export type JobsPanelProps = {
  /** Show dev-only "Run test job" when true (VITE_ENABLE_TEST_JOBS). */
  enableTestJobs?: boolean;
};

export function JobsPanel({ enableTestJobs = false }: JobsPanelProps) {
  const [trackedJobs, setTrackedJobs] = useState<TrackedJob[]>([]);
  const [enqueueing, setEnqueueing] = useState(false);
  const failedNotifiedRef = useRef<Set<string>>(new Set());

  const trackJob = useCallback((jobId: string) => {
    setTrackedJobs((prev) => {
      if (prev.some((j) => j.jobId === jobId)) return prev;
      return [...prev, { jobId, status: 'pending' }];
    });
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
          } catch (err) {
            if (import.meta.env.DEV) {
              console.error('[JobsPanel] poll failed', jobId, err);
            }
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
            if (import.meta.env.DEV) {
              console.error('[JobsPanel] job failed', job);
            }
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
      const msg =
        err instanceof ApiError ? err.message : 'Failed to enqueue test job';
      toast.error(msg);
    } finally {
      setEnqueueing(false);
    }
  };

  return (
    <Card
      title="Jobs"
      size="small"
      style={{ marginBottom: 16 }}
      extra={
        enableTestJobs ? (
          <Button
            size="small"
            icon={<ThunderboltOutlined />}
            loading={enqueueing}
            onClick={() => void handleRunTestJob()}
          >
            Run test job
          </Button>
        ) : null
      }
    >
      {trackedJobs.length === 0 ? (
        <Typography.Text type="secondary">
          {enableTestJobs
            ? 'No jobs yet. Run a test job to verify the worker.'
            : 'Background jobs for this document will appear here.'}
        </Typography.Text>
      ) : (
        <List
          size="small"
          dataSource={trackedJobs}
          renderItem={(job) => (
            <List.Item>
              <Space wrap>
                <Typography.Text code>{shortId(job.jobId)}</Typography.Text>
                {job.type && <Tag>{job.type}</Tag>}
                <Tag color={STATUS_COLORS[job.status]}>{job.status}</Tag>
                {job.status === 'failed' && job.error && (
                  <Typography.Text type="danger">{job.error}</Typography.Text>
                )}
              </Space>
            </List.Item>
          )}
        />
      )}
    </Card>
  );
}
