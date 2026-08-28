import type { DocumentWorkflow } from "../../api/client";
import { DocumentLiveSharingControls } from "../sharing/DocumentLiveSharingControls";

type PageEditorSharingMenuProps = {
  projectId: string;
  documentId: string;
  workflow: DocumentWorkflow;
  publicShareToken: string | null;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  disabled?: boolean;
};

export function PageEditorSharingMenu({
  projectId,
  documentId,
  workflow,
  publicShareToken,
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
        publicShareToken={publicShareToken}
        onWorkflowChange={onWorkflowChange}
        disabled={disabled}
        compact
      />
    </>
  );
}
