import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { toast } from "../components/ui/toast";
import { api, type DocumentWithPartsResponse } from "../api/client";
import { ApiError } from "../api/errors";
import { invalidatePartImage } from "../api/imageCache";
import { resourceTags, invalidateAfter } from "../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { JobsNotice } from "../components/document/JobsNotice";
import { PartList } from "../components/document/PartList";
import { UploadZone } from "../components/document/UploadZone";
import { AppPageShell } from "../components/layout/AppPageShell";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { DocumentLiveSharingPanel } from "../components/sharing/DocumentLiveSharingPanel";
import { WorkflowBadge } from "../components/WorkflowBadge";
import { useFileDrop } from "../hooks/useFileDrop";
import { useServerQuery } from "../hooks/useServerQuery";
import {
  prepareDirectUpload,
  type DirectUploadPayload,
} from "../utils/encodePartImage";
import { renderPdfToPageFiles } from "../utils/pdfToPages";
import { compareFilenames, isPdfFile } from "../utils/uploadBatch";

const ENABLE_TEST_JOBS = process.env.NEXT_PUBLIC_ENABLE_TEST_JOBS === "true";

function formatUpdated(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

type DocumentDetailData = {
  username: string;
  projectName: string;
  document: DocumentWithPartsResponse;
};

export function DocumentDetailPage() {
  const router = useRouter();
  const { projectId, documentId } =
    useParams<{ projectId: string; documentId: string }>() ?? {};
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [reordering, setReordering] = useState(false);
  const [reviewUpdatingPartId, setReviewUpdatingPartId] = useState<
    string | null
  >(null);
  const [titlePanelOpen, setTitlePanelOpen] = useState(false);

  const signedIn = hasAccessToken();
  useEffect(() => {
    if (projectId && documentId && !signedIn) navigateToLogin(router);
  }, [projectId, documentId, signedIn, router]);

  const {
    data,
    loading,
    error,
    refetch: reloadDocument,
    patch: patchDocument,
  } = useServerQuery<DocumentDetailData>({
    key:
      projectId && documentId && signedIn
        ? ["document-detail", projectId, documentId]
        : null,
    tags: [
      resourceTags.currentUser,
      resourceTags.project(projectId ?? ""),
      resourceTags.document(projectId ?? "", documentId ?? ""),
    ],
    read: async () => {
      const [me, project, doc] = await Promise.all([
        api.me(),
        api.getProject(projectId!),
        api.getDocument(projectId!, documentId!),
      ]);
      return {
        username: me.username,
        projectName: project.name,
        document: doc,
      };
    },
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      const msg =
        err instanceof ApiError ? err.message : "Failed to load document";
      toast.error(msg);
      return err instanceof ApiError &&
        (err.status === 403 || err.status === 404)
        ? "This document is not available to your account."
        : msg;
    },
  });

  const document = data?.document ?? null;
  const projectName = data?.projectName ?? null;
  const username = data?.username ?? null;

  const parts = [...(document?.parts ?? [])].sort((a, b) => a.order - b.order);

  const uploadOnePart = async (file: File) => {
    // Prefer the direct-to-storage path: get a presigned Supabase URL, PUT the
    // bytes straight to storage, then finalize. This bypasses Vercel's 4.5 MB
    // function-body cap that a manuscript scan would hit. The payload is never
    // lossy: natively displayable formats upload as the user's original bytes,
    // everything else is transcoded to lossless PNG. If the backend cannot
    // presign (local storage) or the browser cannot decode the file at all,
    // fall back to the legacy multipart upload.
    let payload: DirectUploadPayload | undefined;
    try {
      payload = await prepareDirectUpload(file);
    } catch {
      payload = undefined;
    }

    if (payload) {
      const begin = await api.beginPartUpload(projectId!, documentId!, {
        filename: payload.filename,
        size: payload.blob.size,
      });
      if (begin.upload_url && begin.part_id) {
        const put = await fetch(begin.upload_url, {
          method: "PUT",
          body: payload.blob,
          headers: { "Content-Type": payload.contentType },
        });
        if (!put.ok) {
          throw new Error("Storage upload failed");
        }
        await api.finalizePartUpload(projectId!, documentId!, begin.part_id, {
          image_key: begin.image_key,
          width: payload.width ?? null,
          height: payload.height ?? null,
        });
        return;
      }
      // upload_url is null -> the backend is local storage; use the multipart path.
    }

    await api.uploadPart(projectId!, documentId!, file);
  };

  const handleUpload = async (files: File[]) => {
    if (!projectId || !documentId || files.length === 0) return;
    setUploading(true);
    try {
      // Page order comes from filenames; a PDF holds the slot its name sorts
      // into and expands to its pages in document order.
      const ordered = [...files].sort((a, b) =>
        compareFilenames(a.name, b.name),
      );
      const pages: File[] = [];
      const unreadable: string[] = [];
      for (const file of ordered) {
        if (!isPdfFile(file)) {
          pages.push(file);
          continue;
        }
        try {
          setUploadProgress(`Splitting ${file.name}…`);
          pages.push(
            ...(await renderPdfToPageFiles(file, (done, total) =>
              setUploadProgress(
                `Splitting ${file.name} · page ${done}/${total}`,
              ),
            )),
          );
        } catch {
          unreadable.push(file.name);
        }
      }

      let uploaded = 0;
      const failed: string[] = [];
      let firstFailure: string | null = null;
      for (const [index, file] of pages.entries()) {
        setUploadProgress(
          pages.length > 1
            ? `Uploading page ${index + 1}/${pages.length}`
            : "Uploading…",
        );
        try {
          await uploadOnePart(file);
          uploaded += 1;
        } catch (err) {
          failed.push(file.name);
          firstFailure ??=
            err instanceof ApiError ? err.message : "Upload failed";
        }
      }

      if (uploaded > 0) {
        toast.success(
          uploaded === 1 ? "Part uploaded" : `${uploaded} pages uploaded`,
        );
        invalidateAfter.documentPartsChanged(projectId, documentId);
        await reloadDocument();
      }
      for (const name of unreadable) {
        toast.error(`Could not read ${name} as a PDF`);
      }
      if (failed.length === 1) {
        toast.error(firstFailure ?? "Upload failed");
      } else if (failed.length > 1) {
        toast.error(`Upload failed for ${failed.length} pages`);
      }
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  // Whole-window drop target; the UploadZone button stays as the
  // click-to-pick alternative.
  const dragActive = useFileDrop(
    (files) => void handleUpload(files),
    Boolean(document) && !uploading && !loading,
  );

  const persistOrder = async (partIds: string[]) => {
    if (!projectId || !documentId) return;
    setReordering(true);
    try {
      await api.reorderParts(projectId, documentId, { part_ids: partIds });
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
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
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
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
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
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
            onUpdated={(updated) => {
              patchDocument((current) => ({
                ...current,
                document: {
                  ...current.document,
                  ...(updated.name !== undefined ? { name: updated.name } : {}),
                  ...(updated.workflow !== undefined
                    ? { workflow: updated.workflow }
                    : {}),
                },
              }));
              // The document list, the public view and the copy the page
              // editor fetches all carry these fields; the panel only wrote to
              // the copy this page holds.
              invalidateAfter.documentUpdated(projectId!, documentId!);
            }}
          />
        ) : null
      }
    >
      {loading && !document && !error ? (
        <ContentRegionLoading label="Loading document" />
      ) : (
        <>
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
              progress={uploadProgress}
            />
          )}

          {dragActive && (
            <div className="drop-overlay" role="presentation">
              <div className="drop-overlay__panel">
                <p>Drop to upload</p>
                <p className="hint">
                  Images become pages · a PDF becomes one page per sheet
                </p>
              </div>
            </div>
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
        </>
      )}
    </AppPageShell>
  );
}
