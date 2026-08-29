import { useRef, useState } from "react";
import {
  type DocumentWorkflow,
  type DocumentWorkflowCounts,
} from "../../api/client";
import {
  ActionMenu,
  ActionMenuConfirm,
  ActionMenuItem,
} from "../ui/ActionMenu";
import { DocumentDownloadMenu } from "./DocumentDownloadMenu";
import { DocumentPublishMenu } from "./DocumentPublishMenu";
import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

type DocumentActionBarProps = {
  projectId: string;
  documentId: string;
  documentName: string;
  workflow: DocumentWorkflow;
  publicShareToken: string | null;
  /** Null while the counts read is still in flight. */
  counts: DocumentWorkflowCounts | null;
  publishedPageCount: number;
  /**
   * Publish is owner-only in the backend, so a collaborator is not shown a
   * control whose only possible outcome is a red toast.
   */
  isOwner: boolean;
  uploading: boolean;
  busy: boolean;
  onUpload: (files: File[]) => void;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  onShareTokenChange: (token: string | null) => void;
  onJobsQueued: () => void;
  onOpenSettings: () => void;
  onDeleteDocument: () => Promise<void> | void;
};

/**
 * The one action row a chapter gets.
 *
 * An action lives at the level of the thing it changes. Everything here
 * changes the whole document, so it is here and not in the page editor, which
 * keeps only the acts that change the page open in it.
 */
export function DocumentActionBar({
  projectId,
  documentId,
  documentName,
  workflow,
  publicShareToken,
  counts,
  publishedPageCount,
  isOwner,
  uploading,
  busy,
  onUpload,
  onWorkflowChange,
  onShareTokenChange,
  onJobsQueued,
  onOpenSettings,
  onDeleteDocument,
}: DocumentActionBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const total = counts?.total ?? 0;
  const reviewed = counts?.reviewed ?? 0;

  return (
    <div className="doc-actions" role="group" aria-label="Document actions">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,application/pdf,.pdf"
        multiple
        className="visually-hidden"
        disabled={busy || uploading}
        aria-hidden="true"
        tabIndex={-1}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length > 0) onUpload(files);
          event.target.value = "";
        }}
      />
      <button
        type="button"
        className="btn btn-primary btn-sm"
        disabled={busy || uploading}
        onClick={() => fileInputRef.current?.click()}
      >
        {uploading ? "Uploading…" : "Upload pages"}
      </button>

      <DocumentWorkflowMenu
        projectId={projectId}
        documentId={documentId}
        counts={counts}
        disabled={busy}
        onJobsQueued={onJobsQueued}
      />

      <DocumentDownloadMenu
        projectId={projectId}
        documentId={documentId}
        documentName={documentName}
        counts={counts}
        disabled={busy}
      />

      {isOwner && (
        <DocumentPublishMenu
          projectId={projectId}
          documentId={documentId}
          workflow={workflow}
          publicShareToken={publicShareToken}
          publishedPageCount={publishedPageCount}
          totalPageCount={total}
          reviewedPageCount={reviewed}
          disabled={busy}
          onWorkflowChange={onWorkflowChange}
          onShareTokenChange={onShareTokenChange}
        />
      )}

      <ActionMenu
        label="⋯"
        triggerAriaLabel="More document actions"
        menuLabel="More document actions"
        triggerClassName="btn btn-ghost btn-sm doc-actions__overflow"
        wide
        onOpenChange={(open) => {
          if (!open) setConfirmingDelete(false);
        }}
      >
        {(close) =>
          confirmingDelete ? (
            <ActionMenuConfirm
              destructive
              busy={deleting}
              question={`Delete "${documentName}" and its ${total} pages?`}
              detail="Every page image, segmentation and transcription in this document goes with it. There is no undo."
              confirmLabel={deleting ? "Deleting…" : "Yes, delete the document"}
              onCancel={() => setConfirmingDelete(false)}
              onConfirm={() => {
                setDeleting(true);
                void Promise.resolve(onDeleteDocument()).finally(() => {
                  setDeleting(false);
                  setConfirmingDelete(false);
                  close();
                });
              }}
            />
          ) : (
            <>
              <ActionMenuItem
                label="Document settings"
                detail="Rename the document and see its sharing status."
                onSelect={() => {
                  close();
                  onOpenSettings();
                }}
              />
              <ActionMenuItem
                destructive
                label="Delete document"
                detail="Removes the document and every page in it."
                disabled={busy}
                onSelect={() => setConfirmingDelete(true)}
              />
            </>
          )
        }
      </ActionMenu>
    </div>
  );
}
