import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import type { DocumentPartResponse } from "../../api/client";
import { prefetchPartImage } from "../../api/imageCache";
import { ReviewBadge } from "../WorkflowBadge";
import { AuthenticatedImage } from "../AuthenticatedImage";

type PartListProps = {
  parts: DocumentPartResponse[];
  projectId: string;
  documentId: string;
  loading?: boolean;
  onMoveUp?: (index: number) => void;
  onMoveDown?: (index: number) => void;
  onDelete?: (partId: string) => void;
  onToggleReview?: (partId: string, reviewed: boolean) => void;
  reviewUpdatingPartId?: string | null;
  reordering?: boolean;
};

export function PartList({
  parts,
  projectId,
  documentId,
  loading = false,
  onMoveUp,
  onMoveDown,
  onDelete,
  onToggleReview,
  reviewUpdatingPartId = null,
  reordering = false,
}: PartListProps) {
  if (!loading && parts.length === 0) {
    return <p className="text-muted text-sm">No parts yet. Upload an image.</p>;
  }

  return (
    <div className="part-list" role="list" aria-labelledby="pages-label">
      {parts.map((part, index) => (
        <PartRow
          key={part.id}
          part={part}
          index={index}
          total={parts.length}
          projectId={projectId}
          documentId={documentId}
          onMoveUp={onMoveUp ? () => onMoveUp(index) : undefined}
          onMoveDown={onMoveDown ? () => onMoveDown(index) : undefined}
          onDelete={onDelete ? () => onDelete(part.id) : undefined}
          onToggleReview={
            onToggleReview
              ? (reviewed) => onToggleReview(part.id, reviewed)
              : undefined
          }
          reviewUpdating={reviewUpdatingPartId === part.id}
          reordering={reordering}
        />
      ))}
    </div>
  );
}

type PartRowProps = {
  part: DocumentPartResponse;
  index: number;
  total: number;
  projectId: string;
  documentId: string;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onDelete?: () => void;
  onToggleReview?: (reviewed: boolean) => void;
  reviewUpdating?: boolean;
  reordering?: boolean;
};

function PartRow({
  part,
  index,
  total,
  projectId,
  documentId,
  onMoveUp,
  onMoveDown,
  onDelete,
  onToggleReview,
  reviewUpdating = false,
  reordering,
}: PartRowProps) {
  const router = useRouter();
  const prefetchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dim =
    part.width && part.height
      ? `${part.width} × ${part.height}`
      : "Dimensions pending";
  const editorPath = `/projects/${projectId}/documents/${documentId}/parts/${part.id}`;
  const thumbnailUrl = part.image_url
    ? `${part.image_url}${part.image_url.includes("?") ? "&" : "?"}w=200`
    : null;

  const cancelPrefetch = () => {
    if (prefetchTimer.current !== null) {
      clearTimeout(prefetchTimer.current);
      prefetchTimer.current = null;
    }
  };

  const scheduleFullImagePrefetch = () => {
    if (!part.image_url || prefetchTimer.current !== null) return;
    prefetchTimer.current = setTimeout(() => {
      prefetchTimer.current = null;
      prefetchPartImage(part.image_url);
    }, 100);
  };

  useEffect(
    () => () => {
      if (prefetchTimer.current !== null) clearTimeout(prefetchTimer.current);
    },
    [],
  );

  const openEditor = () => {
    router.push(editorPath);
  };

  const onRowKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openEditor();
    }
  };

  return (
    <article
      className="part-row part-row--clickable"
      role="listitem"
      tabIndex={0}
      aria-label={`Open part ${index + 1} in editor`}
      onClick={openEditor}
      onKeyDown={onRowKeyDown}
      onMouseEnter={scheduleFullImagePrefetch}
      onMouseLeave={cancelPrefetch}
      onFocus={scheduleFullImagePrefetch}
      onBlur={cancelPrefetch}
    >
      <div className="part-thumb" aria-hidden={!!part.image_url}>
        {thumbnailUrl ? (
          <AuthenticatedImage
            compact
            src={thumbnailUrl}
            alt={`Part ${index + 1}`}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              display: "block",
            }}
          />
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth="1.5"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Z"
            />
          </svg>
        )}
      </div>
      <div className="part-info">
        <div className="part-num-row">
          <span className="part-num">Part {index + 1}</span>
          <ReviewBadge reviewed={part.reviewed} />
        </div>
        <div className="part-desc">
          Page {index + 1} of {total}
        </div>
        <div className="part-dim">{dim}</div>
      </div>
      <div className="part-actions" onClick={(e) => e.stopPropagation()}>
        {onToggleReview && (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={reviewUpdating}
            onClick={() => onToggleReview(!part.reviewed)}
            aria-label={
              part.reviewed
                ? `Mark part ${index + 1} unreviewed`
                : `Mark part ${index + 1} reviewed`
            }
          >
            {part.reviewed ? "Unreview" : "Review"}
          </button>
        )}
        {onMoveUp && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            disabled={index === 0 || reordering}
            onClick={onMoveUp}
            aria-label={`Move part ${index + 1} up`}
          >
            ↑
          </button>
        )}
        {onMoveDown && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            disabled={index === total - 1 || reordering}
            onClick={onMoveDown}
            aria-label={`Move part ${index + 1} down`}
          >
            ↓
          </button>
        )}
        {onDelete && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={onDelete}
            aria-label={`Remove part ${index + 1}`}
          >
            ✕
          </button>
        )}
      </div>
    </article>
  );
}
