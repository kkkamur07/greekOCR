import { useBackgroundJobs } from '../../../context/BackgroundJobsContext';

export type { TrackedBackgroundJob as TrackedPageEditorJob } from '../../../context/BackgroundJobsContext';

/** Page-editor alias for the app-wide background job queue. */
export function usePageEditorJobQueue() {
  return useBackgroundJobs();
}
