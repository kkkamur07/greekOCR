import type { DocumentWorkflow } from "../../api/client";
import { DocumentLiveSharingControls } from "../sharing/DocumentLiveSharingControls";

type PageEditorSharingMenuProps = {
  projectId: string;
  documentId: string;
  workflow: DocumentWorkflow;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  disabled?: boolean;
};

export function PageEditorSharingMenu({
  projectId,
  documentId,
  workflow,
  onWorkflowChange,
  disabled = false,
}: PageEditorSharingMenuProps) {
  return (
    <>
      <div className="pe-dd-divider" />
      <div className="pe-dd-section">Sharing</div>
      <DocumentLiveSharingControls
        projectId={projectId}
        documentId={documentId}
        workflow={workflow}
        onWorkflowChange={onWorkflowChange}
        disabled={disabled}
        compact
      />
    </>
  );
}
