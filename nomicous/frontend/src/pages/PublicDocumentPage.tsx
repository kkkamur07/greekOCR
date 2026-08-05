import { useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  api,
  publicPartMediaUrl,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
} from "../api/client";
import { ApiError } from "../api/errors";
import { resourceTags } from "../api/resources";
import { getAccessToken } from "../auth/storage";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { PublicCanvasPdfView } from "../components/public/PublicCanvasPdfView";
import { PublicDocumentExports } from "../components/public/PublicDocumentExports";
import { PublicPageCanvas } from "../components/public/PublicPageCanvas";
import { PublicPartTabs } from "../components/public/PublicPartTabs";
import { PublicTranscriptPanel } from "../components/public/PublicTranscriptPanel";
import { WorkflowBadge } from "../components/WorkflowBadge";
import { useServerQuery } from "../hooks/useServerQuery";
import { linesForPart, publicLinesToRegions } from "../utils/publicLayout";

type PublicDocumentData = {
  document: DocumentWithPartsResponse;
  layout: PublicLayoutResponse;
};

/**
 * A 404 here is not a failure to report but a state of the document - it is not
 * published, or never existed - and gets its own copy, so the two are kept apart
 * rather than collapsed into one message.
 */
type PublicDocumentFailure =
  | { kind: "not-found" }
  | { kind: "message"; text: string };

export function PublicDocumentPage() {
  const { projectId, documentId } =
    useParams<{ projectId: string; documentId: string }>() ?? {};
  const [activePartId, setActivePartId] = useState<string | null>(null);
  const [selectedLineIndex, setSelectedLineIndex] = useState<number | null>(
    null,
  );
  const [canvasView, setCanvasView] = useState<"image" | "pdf">("image");

  const isLoggedIn = !!getAccessToken();

  const { data, loading, error } = useServerQuery<
    PublicDocumentData,
    PublicDocumentFailure
  >({
    key:
      projectId && documentId
        ? ["public-document", projectId, documentId]
        : null,
    tags: [resourceTags.publicDocument(projectId ?? "", documentId ?? "")],
    read: async () => {
      const [document, layout] = await Promise.all([
        api.getPublicDocument(projectId!, documentId!),
        api.getPublicLayout(projectId!, documentId!),
      ]);
      return { document, layout };
    },
    onError: (err) =>
      err instanceof ApiError && err.status === 404
        ? { kind: "not-found" }
        : {
            kind: "message",
            text:
              err instanceof ApiError ? err.message : "Failed to load document",
          },
  });

  const document = data?.document ?? null;
  const layout = data?.layout ?? null;
  const notFound = error?.kind === "not-found";
  const errorMessage = error?.kind === "message" ? error.text : null;

  const parts = useMemo(
    () => [...(document?.parts ?? [])].sort((a, b) => a.order - b.order),
    [document],
  );

  const activePart =
    parts.find((part) => part.id === activePartId) ?? parts[0] ?? null;

  // Settled during render rather than in an effect, so the page never spends a
  // commit disagreeing with the tab strip about which page is open.
  if (activePart && activePart.id !== activePartId) {
    setActivePartId(activePart.id);
  }

  /**
   * `PublicPartTabs` echoes the id it was given back through `onChange` on
   * essentially every render, so only a genuine change of page may clear the
   * reader's line selection - hence the guard rather than an effect keyed on
   * the active id.
   */
  function selectPart(partId: string) {
    if (partId === activePartId) return;
    setActivePartId(partId);
    setSelectedLineIndex(null);
    setCanvasView("image");
  }

  const activePartIndex = activePart
    ? parts.findIndex((part) => part.id === activePart.id) + 1
    : 1;

  const partTabs = parts.map((part, index) => ({
    id: part.id,
    label: `Page ${index + 1}`,
  }));

  const partLines = useMemo(
    () => (activePart ? linesForPart(layout?.lines, activePart.id) : []),
    [layout, activePart],
  );

  const regions = useMemo(() => publicLinesToRegions(partLines), [partLines]);

  const selectedRegionId =
    selectedLineIndex !== null && selectedLineIndex >= 0
      ? selectedLineIndex + 1
      : null;

  const imageUrl = activePart ? publicPartMediaUrl(activePart.id) : null;
  const imageDimensions = {
    width: activePart?.width ?? 0,
    height: activePart?.height ?? 0,
  };

  let content;
  if (loading) {
    content = <ContentRegionLoading label="Loading document" />;
  } else if (notFound) {
    content = (
      <div className="notice-banner" role="alert">
        <strong>Document not available</strong>
        This document is not published or does not exist.
      </div>
    );
  } else if (errorMessage) {
    content = (
      <div className="notice-banner" role="alert">
        <strong>Could not load document</strong>
        {errorMessage}
      </div>
    );
  } else {
    content = (
      <>
        <div className="pub-workspace__toolbar">
          <PublicPartTabs
            parts={partTabs}
            activeId={activePart?.id ?? null}
            onChange={selectPart}
            variant="workspace"
          />

          <div className="pub-workspace__tools">
            <div className="pub-segment" role="tablist" aria-label="Page view">
              <button
                type="button"
                role="tab"
                className={`pub-segment__btn${canvasView === "image" ? " pub-segment__btn--active" : ""}`}
                aria-selected={canvasView === "image"}
                onClick={() => setCanvasView("image")}
              >
                Image
              </button>
              <button
                type="button"
                role="tab"
                className={`pub-segment__btn${canvasView === "pdf" ? " pub-segment__btn--active" : ""}`}
                aria-selected={canvasView === "pdf"}
                onClick={() => setCanvasView("pdf")}
              >
                PDF
              </button>
            </div>

            {projectId && documentId && activePart && (
              <PublicDocumentExports
                projectId={projectId}
                documentId={documentId}
                partId={activePart.id}
                partIndex={activePartIndex}
              />
            )}
          </div>
        </div>

        <div className="pub-split">
          <div
            className="pub-canvas"
            role="img"
            aria-label={
              activePart
                ? `Manuscript page ${activePartIndex}`
                : "Manuscript page"
            }
          >
            {canvasView === "pdf" && projectId && documentId && activePart ? (
              <PublicCanvasPdfView
                projectId={projectId}
                documentId={documentId}
                partId={activePart.id}
              />
            ) : imageUrl && imageDimensions.width > 0 ? (
              <PublicPageCanvas
                imageUrl={imageUrl}
                layoutWidth={imageDimensions.width}
                layoutHeight={imageDimensions.height}
                regions={regions}
                selectedRegionId={selectedRegionId}
                onSelectRegion={(regionId) => {
                  setSelectedLineIndex(regionId === null ? null : regionId - 1);
                }}
              />
            ) : (
              <div className="pub-canvas__empty">
                <p>No page image available</p>
              </div>
            )}
          </div>

          {activePart && (
            <PublicTranscriptPanel
              partId={activePart.id}
              layout={layout}
              selectedLineIndex={selectedLineIndex}
              onSelectLine={setSelectedLineIndex}
            />
          )}
        </div>

        {parts.length === 0 && (
          <p className="list-hint">
            This published document has no page images yet.
          </p>
        )}
      </>
    );
  }

  return (
    <div className="page page--public">
      <nav className="topnav" aria-label="Main navigation">
        <Link href="/" className="topnav-logo" aria-label="nomicous home">
          <img src="/nomos.svg" alt="" />
          <span>nomicous</span>
        </Link>
        <div className="topnav-sep" aria-hidden="true" />
        <div className="topnav-breadcrumb">
          <span className="current" aria-current="page">
            Public view
          </span>
        </div>
        <div className="topnav-spacer" />
        <div className="topnav-actions">
          {isLoggedIn && projectId && documentId && (
            <Link
              href={`/projects/${projectId}/documents/${documentId}`}
              className="btn btn-outline btn-sm"
            >
              Editor
            </Link>
          )}
          {!isLoggedIn && (
            <Link href="/login" className="btn btn-ghost btn-sm">
              Sign in
            </Link>
          )}
        </div>
      </nav>

      <header className="pub-header pub-header--compact">
        <div className="pub-header__main">
          <div className="pub-header__title-row">
            <h1>{document?.name ?? "Document"}</h1>
            {document && <WorkflowBadge workflow={document.workflow} />}
          </div>
          {document && (
            <p className="pub-header__meta">
              {parts.length} page{parts.length === 1 ? "" : "s"}
            </p>
          )}
        </div>
      </header>

      <main className="pub-workspace content-wrap">{content}</main>
    </div>
  );
}
