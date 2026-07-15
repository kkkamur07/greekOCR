import { PageEditorJobProgressPanel } from "./page-editor/PageEditorJobProgressPanel";
import { useBackgroundJobs } from "../context/BackgroundJobsContext";

export function BackgroundJobsPanel() {
  const {
    jobs,
    activeCount,
    panelExpanded,
    setPanelExpanded,
    dismissCompleted,
    cancelJob,
  } = useBackgroundJobs();

  return (
    <PageEditorJobProgressPanel
      jobs={jobs}
      activeCount={activeCount}
      expanded={panelExpanded}
      onExpandedChange={setPanelExpanded}
      onDismissCompleted={dismissCompleted}
      onCancelJob={(jobId) => {
        void cancelJob(jobId);
      }}
    />
  );
}
