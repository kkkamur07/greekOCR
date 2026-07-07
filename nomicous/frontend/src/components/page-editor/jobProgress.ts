import type { JobResponse, JobStatus } from '../../api/client';

export type PageEditorJobKind = 'segmentation' | 'transcription-page' | 'transcription-segment';

export function jobStatusLabel(job: JobResponse): string {
  if (job.status === 'pending') return 'Queued';
  if (job.status === 'running') return 'Starting…';
  if (job.status === 'waiting') return 'Processing…';
  if (job.status === 'done') return 'Complete';
  if (job.status === 'failed') return 'Failed';
  return job.status;
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
