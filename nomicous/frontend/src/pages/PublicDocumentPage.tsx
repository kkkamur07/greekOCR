import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  api,
  publicPartMediaUrl,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
} from '../api/client';
import { ApiError } from '../api/errors';
import { getAccessToken } from '../auth/storage';
import ImageCanvas from '../components/ImageCanvas/ImageCanvas';
import { PublicDocumentDownloads } from '../components/public/PublicDocumentDownloads';
import { PublicPartTabs } from '../components/public/PublicPartTabs';
import { PublicTranscriptPanel } from '../components/public/PublicTranscriptPanel';
import { WorkflowBadge } from '../components/WorkflowBadge';
import {
  linesForPart,
  publicLinesToRegions,
  publicLinesToTranscriptions,
} from '../utils/publicLayout';

export function PublicDocumentPage() {
  const { projectId, documentId } = useParams<{ projectId: string; documentId: string }>();
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(null);
  const [layout, setLayout] = useState<PublicLayoutResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activePartId, setActivePartId] = useState<string | null>(null);
  const [selectedLineIndex, setSelectedLineIndex] = useState<number | null>(null);
  const [pdfPreviewOpen, setPdfPreviewOpen] = useState(false);

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
  const transcriptions = useMemo(
    () => publicLinesToTranscriptions(partLines, null),
    [partLines],
  );

  const selectedRegionId =
    selectedLineIndex !== null && selectedLineIndex >= 0 ? selectedLineIndex + 1 : null;

  const totalLines = layout?.lines?.length ?? 0;
  const imageUrl = activePart ? publicPartMediaUrl(activePart.id) : null;
  const imageDimensions = {
    width: activePart?.width ?? 0,
    height: activePart?.height ?? 0,
  };

  useEffect(() => {
    setSelectedLineIndex(null);
    setPdfPreviewOpen(false);
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
    <div className="page">
      <nav className="topnav" aria-label="Main navigation">
        <Link to="/" className="topnav-logo" aria-label="nomicous home">
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
              to={`/projects/${projectId}/documents/${documentId}`}
              className="btn btn-outline btn-sm"
            >
              Editor
            </Link>
          )}
          {!isLoggedIn && (
            <Link to="/login" className="btn btn-ghost btn-sm">
              Sign in
            </Link>
          )}
        </div>
      </nav>

      <header className="pub-header">
        <div className="pub-header__main">
          <div className="flex items-center gap-2">
            <h1>{document?.name ?? 'Document'}</h1>
            {document && <WorkflowBadge workflow={document.workflow} />}
          </div>
          <p className="meta">
            {parts.length} page{parts.length === 1 ? '' : 's'}
            {activePart && ` · Page ${activePartIndex}: ${partLines.length} line${partLines.length === 1 ? '' : 's'}`}
            {totalLines > 0 && ` · ${totalLines} lines total`}
          </p>
        </div>
        {projectId && documentId && activePart && (
          <PublicDocumentDownloads
            projectId={projectId}
            documentId={documentId}
            partId={activePart.id}
            partIndex={activePartIndex}
            pdfPreviewOpen={pdfPreviewOpen}
            onPdfPreviewOpenChange={setPdfPreviewOpen}
          />
        )}
      </header>

      <PublicPartTabs
        parts={partTabs}
        activeId={activePart?.id ?? null}
        onChange={setActivePartId}
      />

      <main className="content-wrap">
        <div className="pub-split">
          <div
            className="pub-canvas"
            role="img"
            aria-label={
              activePart ? `Manuscript page ${activePartIndex}` : 'Manuscript page'
            }
            style={{ opacity: loading ? 0.6 : 1 }}
          >
            {imageUrl && imageDimensions.width > 0 ? (
              <ImageCanvas
                readOnly
                imageUrl={imageUrl}
                imageDimensions={imageDimensions}
                regions={regions}
                selectedRegionId={selectedRegionId}
                onSelectRegion={(regionId) => {
                  setSelectedLineIndex(regionId === null ? null : regionId - 1);
                }}
                onAddRegion={() => {}}
                onUpdateRegion={() => {}}
                onDeleteRegion={() => {}}
                onTranscribeRegion={() => {}}
              />
            ) : (
              <>
                <p>Manuscript image</p>
                <p className="text-muted text-sm">No page image available</p>
              </>
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

        {transcriptions.length === 0 && partLines.length > 0 && !loading && (
          <p className="list-hint">
            Line geometry is visible on the canvas. Add ground-truth transcriptions in the editor
            to populate this panel.
          </p>
        )}
      </main>
    </div>
  );
}
