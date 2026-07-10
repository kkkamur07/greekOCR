import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import {
  api,
  publicPartMediaUrl,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
} from '../api/client';
import { ApiError } from '../api/errors';
import { getAccessToken } from '../auth/storage';
import { PublicCanvasPdfView } from '../components/public/PublicCanvasPdfView';
import { PublicDocumentExports } from '../components/public/PublicDocumentExports';
import { PublicPageCanvas } from '../components/public/PublicPageCanvas';
import { PublicPartTabs } from '../components/public/PublicPartTabs';
import { PublicTranscriptPanel } from '../components/public/PublicTranscriptPanel';
import { WorkflowBadge } from '../components/WorkflowBadge';
import { linesForPart, publicLinesToRegions } from '../utils/publicLayout';

export function PublicDocumentPage() {
  const { projectId, documentId } = useParams<{ projectId: string; documentId: string }>() ?? {};
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(null);
  const [layout, setLayout] = useState<PublicLayoutResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activePartId, setActivePartId] = useState<string | null>(null);
  const [selectedLineIndex, setSelectedLineIndex] = useState<number | null>(null);
  const [canvasView, setCanvasView] = useState<'image' | 'pdf'>('image');

  const isLoggedIn = !!getAccessToken();

  useEffect(() => {
    if (!projectId || !documentId) return;

    let cancelled = false;
    (async () => {
      setLoading(true);
      setNotFound(false);
      setErrorMessage(null);
      try {
        const [doc, layoutRes] = await Promise.all([
          api.getPublicDocument(projectId, documentId),
          api.getPublicLayout(projectId, documentId),
        ]);
        if (cancelled) return;
        setDocument(doc);
        setLayout(layoutRes);
        const sorted = [...(doc.parts ?? [])].sort((a, b) => a.order - b.order);
        setActivePartId(sorted[0]?.id ?? null);
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true);
          return;
        }
        setErrorMessage(err instanceof ApiError ? err.message : 'Failed to load document');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [projectId, documentId]);

  const parts = useMemo(
    () => [...(document?.parts ?? [])].sort((a, b) => a.order - b.order),
    [document],
  );

  const activePart = parts.find((part) => part.id === activePartId) ?? parts[0] ?? null;
  const activePartIndex = activePart ? parts.findIndex((part) => part.id === activePart.id) + 1 : 1;

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
    selectedLineIndex !== null && selectedLineIndex >= 0 ? selectedLineIndex + 1 : null;

  const imageUrl = activePart ? publicPartMediaUrl(activePart.id) : null;
  const imageDimensions = {
    width: activePart?.width ?? 0,
    height: activePart?.height ?? 0,
  };

  useEffect(() => {
    setSelectedLineIndex(null);
    setCanvasView('image');
  }, [activePartId]);

  if (notFound) {
    return (
      <div className="page">
        <main className="content-wrap">
          <div className="notice-banner" role="alert">
            <strong>Document not available</strong>
            This document is not published or does not exist.
          </div>
        </main>
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="page">
        <main className="content-wrap">
          <div className="notice-banner" role="alert">
            <strong>Could not load document</strong>
            {errorMessage}
          </div>
        </main>
      </div>
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
            <h1>{document?.name ?? 'Document'}</h1>
            {document && <WorkflowBadge workflow={document.workflow} />}
          </div>
          <p className="pub-header__meta">
            {parts.length} page{parts.length === 1 ? '' : 's'}
          </p>
        </div>
      </header>

      <main className="pub-workspace content-wrap">
        <div className="pub-workspace__toolbar">
          <PublicPartTabs
            parts={partTabs}
            activeId={activePart?.id ?? null}
            onChange={setActivePartId}
            variant="workspace"
          />

          <div className="pub-workspace__tools">
            <div className="pub-segment" role="tablist" aria-label="Page view">
              <button
                type="button"
                role="tab"
                className={`pub-segment__btn${canvasView === 'image' ? ' pub-segment__btn--active' : ''}`}
                aria-selected={canvasView === 'image'}
                onClick={() => setCanvasView('image')}
              >
                Image
              </button>
              <button
                type="button"
                role="tab"
                className={`pub-segment__btn${canvasView === 'pdf' ? ' pub-segment__btn--active' : ''}`}
                aria-selected={canvasView === 'pdf'}
                onClick={() => setCanvasView('pdf')}
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

        <div className="pub-split" style={{ opacity: loading ? 0.6 : 1 }}>
          <div
            className="pub-canvas"
            role="img"
            aria-label={
              activePart ? `Manuscript page ${activePartIndex}` : 'Manuscript page'
            }
          >
            {canvasView === 'pdf' && projectId && documentId && activePart ? (
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

        {!loading && parts.length === 0 && (
          <p className="list-hint">This published document has no page images yet.</p>
        )}
      </main>
    </div>
  );
}
