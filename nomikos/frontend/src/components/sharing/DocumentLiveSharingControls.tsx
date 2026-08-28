import { useState } from "react";
import Link from "next/link";
import { api, type DocumentWorkflow } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";
import { WorkflowBadge } from "../WorkflowBadge";
import {
  publicDocumentPath,
  publicDocumentUrl,
} from "../../utils/publicDocumentUrl";

type DocumentLiveSharingControlsProps = {
  projectId: string;
  documentId: string;
  workflow: DocumentWorkflow;
  /**
   * Set only in the owner-facing document response - a collaborator's copy
   * carries `null` here even once the document is published, since the token
   * is what makes the link work and only the owner may hand it out.
   */
  publicShareToken: string | null;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  disabled?: boolean;
  compact?: boolean;
};

export function DocumentLiveSharingControls({
  projectId,
  documentId,
  workflow,
  publicShareToken,
  onWorkflowChange,
  disabled = false,
  compact = false,
}: DocumentLiveSharingControlsProps) {
  const [publishing, setPublishing] = useState(false);
  const isPublished = workflow === "published";
  const isArchived = workflow === "archived";
  const publicPath = publicShareToken
    ? publicDocumentPath(projectId, documentId, publicShareToken)
    : null;
  const publicUrl = publicShareToken
    ? publicDocumentUrl(projectId, documentId, publicShareToken)
    : null;
  const busy = disabled || publishing;

  async function handlePublishToggle() {
    if (isArchived) return;
    setPublishing(true);
    try {
      const nextWorkflow: DocumentWorkflow = isPublished
        ? "draft"
        : "published";
      const updated = await api.updateDocument(projectId, documentId, {
        workflow: nextWorkflow,
      });
      onWorkflowChange(updated.workflow);
      toast.success(
        nextWorkflow === "published"
          ? "Document published. Public link is live"
          : "Document returned to draft",
      );
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Failed to update document status";
      toast.error(message);
    } finally {
      setPublishing(false);
    }
  }

  async function handleCopyPublicLink() {
    if (!publicUrl) return;
    try {
      await navigator.clipboard.writeText(publicUrl);
      toast.success("Public link copied");
    } catch {
      toast.error("Could not copy link");
    }
  }

  const statusClass = compact
    ? "entity-panel__meta"
    : "entity-panel__status-row";
  const urlClass = compact ? "pe-dd-share__url" : "entity-panel__url";
  const actionsClass = compact
    ? "pe-dd-share__actions"
    : "entity-panel__actions";

  return (
    <>
      <div className={statusClass}>
        {!compact && <span className="entity-panel__status-label">Status</span>}
        <WorkflowBadge workflow={workflow} />
      </div>
      {isPublished && publicUrl && publicPath && (
        <div className={compact ? "pe-dd-share" : "entity-panel__share-block"}>
          <label
            className="entity-panel__label"
            htmlFor={`public-url-${documentId}`}
          >
            Public page
          </label>
          <input
            id={`public-url-${documentId}`}
            className={urlClass}
            type="text"
            readOnly
            value={publicUrl}
            aria-label="Public document URL"
          />
          <div className={actionsClass}>
            <button
              type="button"
              className="btn btn-outline btn-xs"
              disabled={busy}
              onClick={() => void handleCopyPublicLink()}
            >
              Copy link
            </button>
            <Link
              href={publicPath}
              className="btn btn-ghost btn-xs"
              target="_blank"
              rel="noopener noreferrer"
            >
              Open public view
            </Link>
          </div>
        </div>
      )}
      {/*
        A collaborator's document response never carries the token - only the
        owner may hand out the link - so there is nothing here to build. Say
        so plainly rather than showing a link that 404s for whoever opens it.
      */}
      {isPublished && !publicShareToken && (
        <p className={compact ? "pe-dd-share" : "entity-panel__hint"}>
          Only the project owner can get the public share link.
        </p>
      )}
      {!isArchived &&
        (compact ? (
          <button
            type="button"
            role="menuitem"
            className="pe-dd-item"
            disabled={busy}
            onClick={() => void handlePublishToggle()}
          >
            {publishing
              ? "Updating…"
              : isPublished
                ? "Unpublish document"
                : "Publish live page"}
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={busy}
            onClick={() => void handlePublishToggle()}
          >
            {publishing
              ? "Updating…"
              : isPublished
                ? "Unpublish document"
                : "Publish live page"}
          </button>
        ))}
    </>
  );
}
