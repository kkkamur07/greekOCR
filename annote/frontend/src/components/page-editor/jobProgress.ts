import type { JobResponse, JobStatus } from '../../api/client';

export type PageEditorJobKind = 'segmentation' | 'transcription-page' | 'transcription-segment';

export function jobStatusLabel(job: JobResponse): string {
  if (job.status === 'pending') return 'Queued';
  if (job.status === 'running') return 'Starting…';
  if (job.status === 'waiting') {
    const transcribeProgress = transcribeLineProgress(job);
    return transcribeProgress ?? 'Processing…';
  }
  if (job.status === 'done') return 'Complete';
  if (job.status === 'failed') return 'Failed';
  return job.status;
}

export function transcribeLineProgress(job: JobResponse): string | null {
  if (job.type !== 'transcribe') return null;
  const payload = job.payload ?? {};
  const lineJobs = Array.isArray(payload.ml_line_jobs) ? payload.ml_line_jobs : [];
  if (lineJobs.length === 0) return null;
  const outputs =
    payload.ml_line_outputs && typeof payload.ml_line_outputs === 'object'
      ? Object.keys(payload.ml_line_outputs as Record<string, unknown>).length
      : 0;
  return `Transcribing ${outputs}/${lineJobs.length} segment${lineJobs.length === 1 ? '' : 's'}`;
}

export function isTerminalJobStatus(status: JobStatus): boolean {
  return status === 'done' || status === 'failed';
}

export function pageEditorJobKindLabel(kind: PageEditorJobKind): string {
  switch (kind) {
    case 'segmentation':
      return 'Segmentation';
    case 'transcription-page':
      return 'Page OCR';
    case 'transcription-segment':
      return 'Segment OCR';
    default:
      return 'Job';
  }
}
