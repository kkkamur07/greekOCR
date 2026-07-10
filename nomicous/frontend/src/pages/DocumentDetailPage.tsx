import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { toast } from "../components/ui/toast";
import { api, type DocumentWithPartsResponse } from "../api/client";
import { ApiError } from "../api/errors";
import { invalidatePartImage } from "../api/imageCache";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { JobsNotice } from "../components/document/JobsNotice";
import { PartList } from "../components/document/PartList";
import { UploadZone } from "../components/document/UploadZone";
import { AppPageShell } from "../components/layout/AppPageShell";
import { DocumentLiveSharingPanel } from "../components/sharing/DocumentLiveSharingPanel";
import { WorkflowBadge } from "../components/WorkflowBadge";

const ENABLE_TEST_JOBS = process.env.NEXT_PUBLIC_ENABLE_TEST_JOBS === "true";

function formatUpdated(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function DocumentDetailPage() {
  const router = useRouter();
  const { projectId, documentId } =
    useParams<{ projectId: string; documentId: string }>() ?? {};
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(
    null,
  );
  const [projectName, setProjectName] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reordering, setReordering] = useState(false);
  const [reviewUpdatingPartId, setReviewUpdatingPartId] = useState<
    string | null
  >(null);
  const [titlePanelOpen, setTitlePanelOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!projectId || !documentId) return;
    if (!hasAccessToken()) {
      navigateToLogin(router);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [me, proj, doc] = await Promise.all([
        api.me(),
        api.getProject(projectId),
        api.getDocument(projectId, documentId),
      ]);
      setUsername(me.username);
      setProjectName(proj.name);
      setDocument(doc);
    } catch (err) {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return;
      }
      const msg =
        err instanceof ApiError ? err.message : "Failed to load document";
      setDocument(null);
      setError(
        err instanceof ApiError && (err.status === 403 || err.status === 404)
          ? "This document is not available to your account."
          : msg,
      );
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }, [projectId, documentId, router]);

  useEffect(() => {
    void load();
  }, [load]);

  const parts = [...(document?.parts ?? [])].sort((a, b) => a.order - b.order);

  const handleUpload = async (file: File) => {
    if (!projectId || !documentId) return;
    setUploading(true);
    try {
      await api.uploadPart(projectId, documentId, file);
      toast.success("Part uploaded");
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Upload failed";
      toast.error(msg);
    } finally {
      setUploading(false);
    }
  };

  const persistOrder = async (partIds: string[]) => {
    if (!projectId || !documentId) return;
    setReordering(true);
    try {
      await api.reorderParts(projectId, documentId, { part_ids: partIds });
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Reorder failed";
      toast.error(msg);
    } finally {
      setReordering(false);
    }
  };

  const movePart = (index: number, direction: -1 | 1) => {
    const next = index + direction;
    if (next < 0 || next >= parts.length) return;
    const ids = parts.map((p) => p.id);
    [ids[index], ids[next]] = [ids[next], ids[index]];
    void persistOrder(ids);
  };

  const handleDelete = async (partId: string) => {
    if (!projectId || !documentId) return;
    try {
      await api.deletePart(projectId, documentId, partId);
      invalidatePartImage(partId);
      toast.success("Part removed");
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Delete failed";
      toast.error(msg);
    }
  };

  const handleToggleReview = async (partId: string, reviewed: boolean) => {
    if (!projectId || !documentId) return;
    setReviewUpdatingPartId(partId);
    try {
      await api.updatePartReviewStatus(projectId, documentId, partId, {
        reviewed,
      });
      toast.success(
        reviewed ? "Part marked reviewed" : "Part marked unreviewed",
      );
      await load();
    } catch (err) {
      const msg =
        err instanceof ApiError && err.status === 403
          ? "Only project members can change review status."
          : err instanceof ApiError
            ? err.message
            : "Review status update failed";
      toast.error(msg);
    } finally {
      setReviewUpdatingPartId(null);
    }
  };

  const subtitle = document
    ? `${parts.length} part${parts.length === 1 ? "" : "s"} · updated ${formatUpdated(document.updated_at)}`
    : undefined;

  return (
    <AppPageShell
      breadcrumb={[
        { label: "Projects", href: "/projects" },
        {
          label: projectName ?? "Project",
          href: projectId ? `/projects/${projectId}` : undefined,
        },
        { label: document?.name ?? "Document" },
      ]}
      username={username}
      title={document?.name ?? "Document"}
      subtitle={subtitle}
      titleExtra={
        document ? <WorkflowBadge workflow={document.workflow} /> : undefined
      }
      titleEditable={Boolean(document && projectId && documentId)}
      titlePanelOpen={titlePanelOpen}
      onTitlePanelToggle={() => setTitlePanelOpen((open) => !open)}
      titlePanelLabel="Document settings and live sharing"
      titlePanel={
        document && projectId && documentId ? (
          <DocumentLiveSharingPanel
            projectId={projectId}
            documentId={documentId}
            name={document.name}
            workflow={document.workflow}
            onUpdated={(patch) => {
              setDocument((current) =>
                current
                  ? {
                      ...current,
                      ...(patch.name !== undefined ? { name: patch.name } : {}),
                      ...(patch.workflow !== undefined
                        ? { workflow: patch.workflow }
                        : {}),
                    }
                  : current,
              );
            }}
          />
        ) : null
      }
    >
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Document unavailable</strong>
          {error}
        </div>
      )}

      {document && <JobsNotice enableTestJobs={ENABLE_TEST_JOBS} />}

      {document && (
        <UploadZone
          onUpload={handleUpload}
          disabled={loading}
          loading={uploading}
        />
      )}

      {document && (
        <>
          <p className="section-label" id="pages-label">
            Pages
          </p>
          <PartList
            parts={parts}
            projectId={projectId!}
            documentId={documentId!}
            loading={loading}
            onMoveUp={(i) => movePart(i, -1)}
            onMoveDown={(i) => movePart(i, 1)}
            onDelete={(id) => void handleDelete(id)}
            onToggleReview={(partId, reviewed) =>
              void handleToggleReview(partId, reviewed)
            }
            reviewUpdatingPartId={reviewUpdatingPartId}
            reordering={reordering}
          />
        </>
      )}
    </AppPageShell>
  );
}
