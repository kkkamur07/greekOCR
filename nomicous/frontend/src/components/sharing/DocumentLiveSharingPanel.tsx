import { useEffect, useState } from "react";
import { api, type DocumentWorkflow } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";
import { DocumentLiveSharingControls } from "./DocumentLiveSharingControls";

type DocumentLiveSharingPanelProps = {
  projectId: string;
  documentId: string;
  name: string;
  workflow: DocumentWorkflow;
  onUpdated: (patch: { name?: string; workflow?: DocumentWorkflow }) => void;
};

export function DocumentLiveSharingPanel({
  projectId,
  documentId,
  name,
  workflow,
  onUpdated,
}: DocumentLiveSharingPanelProps) {
  const [draftName, setDraftName] = useState(name);
  const [savingName, setSavingName] = useState(false);

  useEffect(() => {
    setDraftName(name);
  }, [name]);

  async function handleSaveName() {
    const trimmed = draftName.trim();
    if (!trimmed || trimmed === name) return;
    setSavingName(true);
    try {
      const updated = await api.updateDocument(projectId, documentId, {
        name: trimmed,
      });
      onUpdated({ name: updated.name });
      toast.success("Document renamed");
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Failed to rename document";
      toast.error(message);
    } finally {
      setSavingName(false);
    }
  }

  const nameChanged = draftName.trim() !== name && draftName.trim().length > 0;

  return (
    <>
      <div className="entity-panel__section">
        <h2 className="entity-panel__heading">Document</h2>
        <div className="field">
          <label htmlFor="entity-document-name">Name</label>
          <input
            id="entity-document-name"
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void handleSaveName();
              }
            }}
          />
        </div>
        <button
          type="button"
          className="btn btn-outline btn-sm"
          disabled={!nameChanged || savingName}
          onClick={() => void handleSaveName()}
        >
          {savingName ? "Saving…" : "Save name"}
        </button>
      </div>
      <div className="entity-panel__section">
        <h2 className="entity-panel__heading">Live sharing</h2>
        <p className="entity-panel__hint">
          Publish this document as a read-only public page for collaborators and
          readers.
        </p>
        <DocumentLiveSharingControls
          projectId={projectId}
          documentId={documentId}
          workflow={workflow}
          onWorkflowChange={(nextWorkflow) =>
            onUpdated({ workflow: nextWorkflow })
          }
        />
      </div>
    </>
  );
}
