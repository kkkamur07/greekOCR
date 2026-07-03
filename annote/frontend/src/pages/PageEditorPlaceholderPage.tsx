import { useState, type CSSProperties, type MouseEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Alert, Button, Input, Space, Spin, Typography } from 'antd';
import {
  api,
  type LayoutPoint,
  type LinePoint,
  type LineResponse,
  type PartLayoutResponse,
} from '../api/client';
import { ApiError } from '../api/errors';
import { AuthenticatedImage } from '../components/AuthenticatedImage';
import {
  useLayoutMutations,
  usePageEditorData,
  usePairingState,
} from '../components/page-editor/hooks';

function reviewMutationMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return 'Only project members can change Review status.';
  }
  return error instanceof Error ? error.message : 'Review status update failed.';
}

function editorButton(active: boolean): string {
  return active ? 'btn btn--primary btn--sm' : 'btn btn--ghost btn--sm';
}

const toolbarClusterStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  paddingInline: 4,
  borderLeft: '1px solid #e5e7eb',
} satisfies CSSProperties;

export function PageEditorPlaceholderPage() {
  const { projectId, documentId, partId } = useParams<{
    projectId: string;
    documentId: string;
    partId: string;
  }>();

  const [editorMode, setEditorMode] = useState<'layout' | 'transcription'>('layout');
  const [drawMode, setDrawMode] = useState<'none' | 'rectangle' | 'polygon'>('none');
  const [draftStart, setDraftStart] = useState<LayoutPoint | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<LinePoint[]>([]);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [actionsOpen, setActionsOpen] = useState(false);

  const editorData = usePageEditorData(projectId, documentId, partId, () => {
    setEditorMode('layout');
    setDrawMode('none');
    setDraftPolygon([]);
    setDraftStart(null);
    setReviewError(null);
  });
  const {
    document,
    setDocument,
    part,
    setPart,
    layout,
    setLayout,
    lines,
    setLines,
    loading,
    error,
    layoutError,
    lineError,
    setLineError,
    transcriptionLayers,
    setTranscriptionLayers,
    selectedTranscriptionLayerId,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    textLines,
    setTextLines,
    pairingProgress,
    setPairingProgress,
    pairingError,
    setPairingError,
    transcribeModels,
    selectedTranscribeModelId,
    setSelectedTranscribeModelId,
    partIndex,
  } = editorData;

  const pairing = usePairingState({
    projectId,
    documentId,
    partId,
    lines,
    setLines,
    transcriptionLayers,
    setTranscriptionLayers,
    selectedTranscriptionLayerId,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    setTextLines,
    setPairingProgress,
    setPairingError,
    selectedTranscribeModelId,
  });

  const layoutMutations = useLayoutMutations({
    projectId,
    documentId,
    partId,
    layout,
    setLayout,
    lines,
    setLines,
    setLineError,
    setTextLines,
    setPairingProgress,
    setPairingError,
    selectedSegmentId: pairing.selectedSegmentId,
    setSelectedSegmentId: pairing.setSelectedSegmentId,
    setApprovedTextDraft: pairing.setApprovedTextDraft,
    onDrawComplete: () => setDrawMode('none'),
  });

  const {
    selectedLineId,
    setSelectedLineId,
    setSelectedLineSnapshot,
    saveMessage,
    setSaveMessage,
    mutationError,
    segmenting,
    useOtsuRefinement,
    setUseOtsuRefinement,
    segmentMessage,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
    replaceWithManualLine,
    moveSelectedSegmentRight,
    deleteSelectedSegment,
    runAutoSegment,
  } = layoutMutations;

  const {
    selectedSegmentId,
    setSelectedSegmentId,
    pageTranscriptionText,
    setPageTranscriptionText,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    copyMessage,
    ocrRunning,
    ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    lineTextForLayer,
    importPageTranscription,
    pairTextLine,
    saveApprovedText,
    selectTranscriptionLayer,
    saveGroundTruthText,
    runSegmentOcr,
    runPageOcr,
    copySelectedLayerToGroundTruth,
    selectSegment,
  } = pairing;

  function pickDrawMode(nextMode: 'rectangle' | 'polygon') {
    setDrawMode((mode) => (mode === nextMode ? 'none' : nextMode));
    setDraftPolygon([]);
    setDraftStart(null);
    setActionsOpen(false);
  }

  async function updateReviewStatus(reviewed: boolean) {
    if (!projectId || !documentId || !partId) return;
    try {
      const updated = await api.updatePartReviewStatus(projectId, documentId, partId, {
        reviewed,
      });
      setPart(updated);
      setDocument((current) =>
        current
          ? {
              ...current,
              parts: current.parts.map((item) => (item.id === updated.id ? updated : item)),
            }
          : current,
      );
      setReviewError(null);
    } catch (err) {
      setReviewError(reviewMutationMessage(err));
    }
  }

  if (loading) {
    return (
      <div style={{ padding: 24 }}>
        <Space>
          <Spin />
          <Typography.Text>Loading page...</Typography.Text>
        </Space>
      </div>
    );
  }

  if (error || !document || !part) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          type="warning"
          showIcon
          message="Page unavailable"
          description={error ?? 'This document part was not found.'}
        />
      </div>
    );
  }

  const statusAlerts = (
    <div style={{ display: 'grid', gap: 8 }}>
      {saveMessage && <Alert type="success" showIcon message={saveMessage} />}
      {transcriptionSaveMessage && <Alert type="success" showIcon message={transcriptionSaveMessage} />}
      {copyMessage && <Alert type="success" showIcon message={copyMessage} />}
      {ocrMessage && <Alert type="success" showIcon message={ocrMessage} />}
      {segmentMessage && <Alert type="success" showIcon message={segmentMessage} />}
      {mutationError && <Alert type="error" showIcon message={mutationError} />}
      {pairingError && <Alert type="warning" showIcon message={pairingError} />}
      {reviewError && <Alert type="warning" showIcon message={reviewError} />}
      {layoutError && (
        <Alert type="warning" showIcon message="Layout API unavailable" description={layoutError} />
      )}
      {lineError && (
        <Alert type="warning" showIcon message="Segment API unavailable" description={lineError} />
      )}
    </div>
  );

  return (
    <div style={{ display: 'flex', height: '100vh', flexDirection: 'column', overflow: 'hidden', background: '#fff' }}>
      <header
        style={{
          display: 'flex',
          flexShrink: 0,
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          borderBottom: '1px solid #e5e7eb',
          padding: '8px 12px',
        }}
      >
        <span className="visually-hidden">ANNOTE PAGE WORKSPACE</span>
        <h2 className="visually-hidden">{editorMode === 'layout' ? 'Layout edit' : 'Transcription edit'}</h2>
        <span className="visually-hidden">
          Review status: {part.reviewed ? 'Reviewed' : 'Unreviewed'}
        </span>
        {!selectedSegment && editorMode !== 'transcription' && (
          <span className="visually-hidden">
            Pairing progress: {pairingProgress.paired_lines}/{pairingProgress.total_lines} Lines paired
          </span>
        )}
        <span className="visually-hidden">
          {lines.length} {lines.length === 1 ? 'Segment' : 'Segments'}
        </span>
        <div className="visually-hidden">
          <label>
            Page transcription text
            <textarea
              aria-label="Page transcription text"
              value={pageTranscriptionText}
              onChange={(event) => setPageTranscriptionText(event.target.value)}
            />
          </label>
          <button type="button" onClick={() => void importPageTranscription()}>
            Import page transcription
          </button>
          {!selectedSegmentId && textLines.map((textLine) => {
            const pairedIndex = textLine.paired_line_id
              ? [...lines]
                  .sort((a, b) => a.order - b.order)
                  .findIndex((line) => line.id === textLine.paired_line_id)
              : -1;
            const pairedLabel = pairedIndex >= 0 ? ` · paired with Segment ${pairedIndex + 1}` : '';
            return (
              <div key={textLine.order}>
                <span>
                  Text line {textLine.order + 1}
                  {pairedLabel}
                </span>
                <button
                  type="button"
                  disabled={!selectedSegmentId}
                  onClick={() => void pairTextLine(textLine.order)}
                >
                  Pair Text line {textLine.order + 1}
                </button>
              </div>
            );
          })}
        </div>
        <div style={{ display: 'flex', minWidth: 0, alignItems: 'center', gap: 8 }}>
          {projectId && documentId && (
            <Link
              to={`/projects/${projectId}/documents/${documentId}`}
              aria-label="Document parts"
              style={{ flexShrink: 0, color: '#6b7280', fontSize: 14 }}
            >
              ← Back
            </Link>
          )}
          <h1
            title={`${document.name} · Page ${partIndex}`}
            style={{ margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 16, fontWeight: 500 }}
          >
            {document.name} · Page {partIndex}
          </h1>
          <span style={{ color: '#92400e', fontSize: 12 }}>
            {part.reviewed ? 'reviewed' : 'unreviewed'}
          </span>
        </div>

        <div style={{ display: 'flex', minWidth: 0, flex: 1, justifyContent: 'center', paddingInline: 8 }}>
          <div style={{ color: '#6b7280', fontSize: 13 }}>
            Pairing {pairingProgress.paired_lines}/{pairingProgress.total_lines} · {lines.length}{' '}
            {lines.length === 1 ? 'segment' : 'segments'}
          </div>
        </div>

        <div style={{ display: 'flex', flexShrink: 0, alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
          <button type="button" onClick={() => { setDrawMode('none'); setActionsOpen(false); }} className={editorButton(drawMode === 'none' && editorMode === 'layout')}>
            Pan
          </button>
          <button type="button" onClick={() => { setDrawMode('none'); setActionsOpen(false); }} className={editorButton(drawMode === 'none' && editorMode === 'layout')}>
            Select
          </button>
          <button
            type="button"
            aria-label="Rectangle segment"
            onClick={() => pickDrawMode('rectangle')}
            className={editorButton(drawMode === 'rectangle')}
          >
            Rect
          </button>
          <button
            type="button"
            aria-label="Polygon segment"
            onClick={() => pickDrawMode('polygon')}
            className={editorButton(drawMode === 'polygon')}
          >
            Poly
          </button>
          <button
            type="button"
            aria-label="Transcription edit"
            onClick={() => {
              setEditorMode((mode) => (mode === 'transcription' ? 'layout' : 'transcription'));
              setDrawMode('none');
              setActionsOpen(false);
            }}
            className={editorButton(editorMode === 'transcription')}
          >
            Text
          </button>
          <div style={toolbarClusterStyle}>
            <button
              type="button"
              aria-label="Move segment right"
              disabled={!selectedSegmentId}
              onClick={() => void moveSelectedSegmentRight()}
              className={editorButton(false)}
            >
              Move
            </button>
            <button
              type="button"
              aria-label={selectedSegmentId ? 'Delete Segment' : 'Delete'}
              disabled={!selectedSegmentId && !selectedLineId}
              onClick={() => {
                if (selectedSegmentId) void deleteSelectedSegment();
                if (selectedLineId) void resetSelectedLine();
              }}
              className="btn btn--danger-ghost btn--sm"
            >
              Del
            </button>
          </div>
          <div style={toolbarClusterStyle}>
            <button type="button" className={editorButton(true)}>
              Lines
            </button>
            <button type="button" disabled className={editorButton(false)}>
              Ceiling
            </button>
          </div>
          <div style={{ ...toolbarClusterStyle, position: 'relative' }}>
            <button
              type="button"
              aria-haspopup="menu"
              aria-expanded={actionsOpen}
              onClick={() => setActionsOpen((open) => !open)}
              className="btn btn--ghost btn--sm"
            >
              Process
            </button>
            {actionsOpen && (
              <div
                role="menu"
                aria-label="Page processing actions"
                style={{
                  position: 'absolute',
                  top: 'calc(100% + 6px)',
                  right: 0,
                  zIndex: 20,
                  display: 'grid',
                  gap: 6,
                  width: 240,
                  padding: 8,
                  border: '1px solid #d1d5db',
                  borderRadius: 8,
                  background: '#fff',
                  boxShadow: '0 12px 32px rgba(15, 23, 42, 0.18)',
                }}
              >
                <div style={{ padding: '2px 6px', fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Segmentation
                </div>
                <button type="button" role="menuitem" disabled className="btn btn--ghost btn--sm">
                  Binarize
                </button>
                <label
                  role="menuitem"
                  className="btn btn--ghost btn--sm"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}
                >
                  <input
                    type="checkbox"
                    aria-label="Refine Kraken segments with Otsu"
                    checked={useOtsuRefinement}
                    disabled={segmenting}
                    onChange={(event) => setUseOtsuRefinement(event.target.checked)}
                  />
                  Otsu refine
                </label>
                <button
                  type="button"
                  role="menuitem"
                  disabled={segmenting || ocrRunning}
                  onClick={() => {
                    setActionsOpen(false);
                    void runAutoSegment();
                  }}
                  className="btn btn--ghost btn--sm"
                >
                  {segmenting ? 'Segmenting…' : 'Auto segment'}
                </button>
                <div style={{ height: 1, background: '#e5e7eb', marginBlock: 2 }} />
                <div style={{ padding: '2px 6px', fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  OCR
                </div>
                {transcribeModels.length > 0 && (
                  <label className="btn btn--ghost btn--sm" style={{ display: 'inline-flex', justifyContent: 'space-between', gap: 8 }}>
                    <span>OCR model</span>
                    <select
                      aria-label="OCR model"
                      value={selectedTranscribeModelId ?? ''}
                      disabled={ocrRunning}
                      onChange={(event) => setSelectedTranscribeModelId(event.target.value || null)}
                      style={{ minWidth: 0, flex: 1 }}
                    >
                      {transcribeModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <button
                  type="button"
                  role="menuitem"
                  disabled={!selectedSegmentId || ocrRunning || segmenting || !selectedTranscribeModelId}
                  onClick={() => {
                    setActionsOpen(false);
                    void runSegmentOcr();
                  }}
                  className="btn btn--ghost btn--sm"
                >
                  {ocrRunning ? 'OCR…' : 'OCR segment'}
                </button>
                <button
                  type="button"
                  role="menuitem"
                  disabled={ocrRunning || segmenting || lines.length === 0 || !selectedTranscribeModelId}
                  onClick={() => {
                    setActionsOpen(false);
                    void runPageOcr();
                  }}
                  className="btn btn--ghost btn--sm"
                >
                  {ocrRunning ? 'OCR…' : 'OCR page'}
                </button>
              </div>
            )}
          </div>
          <div style={toolbarClusterStyle}>
            <button type="button" onClick={() => void updateReviewStatus(!part.reviewed)} className="btn btn--ghost btn--sm">
              {part.reviewed ? 'Mark unreviewed' : 'Mark reviewed'}
            </button>
            <button type="button" disabled className="btn btn--primary btn--sm">
              Export
            </button>
          </div>
        </div>
      </header>

      {(saveMessage || transcriptionSaveMessage || copyMessage || ocrMessage || segmentMessage || mutationError || pairingError || reviewError || layoutError || lineError) && (
        <div style={{ flexShrink: 0, borderBottom: '1px solid #e5e7eb', padding: 8 }}>{statusAlerts}</div>
      )}

      <main style={{ display: 'flex', minHeight: 0, flex: 1, flexDirection: 'column' }}>
        <div
          style={{
            position: 'relative',
            minHeight: 0,
            flex: 1,
            overflow: 'hidden',
            background: '#f5f5f5',
          }}
        >
          <LayoutCanvas
            imageUrl={part.image_url}
            imageAlt={`Page ${partIndex}`}
            imageWidth={part.width ?? 640}
            imageHeight={part.height ?? 900}
            layout={layout}
            lines={lines}
            drawingRectangle={drawMode === 'rectangle'}
            drawingPolygon={drawMode === 'polygon'}
            onDraftStart={setDraftStart}
            onRectangleDrawn={async (end) => {
              if (!draftStart) return;
              const rectangle = rectanglePoints(draftStart, end);
              await replaceWithManualLine('rectangle', rectangle);
              setDraftStart(null);
            }}
            onPolygonPoint={(point) => setDraftPolygon((current) => [...current, point])}
            onPolygonComplete={async () => {
              if (draftPolygon.length >= 3) {
                await replaceWithManualLine('polygon', draftPolygon);
              }
              setDraftPolygon([]);
            }}
            onSelectLine={(lineId) => {
              const selectedLine = layout.lines.find((line) => line.id === lineId);
              setSelectedLineId(lineId);
              setSelectedSegmentId(null);
              setSelectedLineSnapshot({
                baseline: selectedLine?.baseline,
                mask: selectedLine?.mask,
              });
            }}
            onSelectSegment={(lineId) => {
              setSelectedLineId(null);
              setSaveMessage(null);
              selectSegment(lineId);
            }}
          />
        </div>

        {selectedLineId && (
          <div style={{ flexShrink: 0, borderTop: '1px solid #e5e7eb', background: '#fff', padding: 12 }}>
            <Space wrap>
              <Typography.Text>Selected baseline</Typography.Text>
              <Button onClick={() => moveSelectedBaseline(5)}>Move baseline down</Button>
              <Button type="primary" onClick={() => void saveSelectedLine()}>
                Save layout
              </Button>
              <Button danger onClick={() => void resetSelectedLine()}>
                Reset layout
              </Button>
            </Space>
          </div>
        )}

        {(selectedSegment || editorMode === 'transcription') && (
          <div style={{ flexShrink: 0, borderTop: '1px solid #e5e7eb', background: '#fff', padding: 12 }}>
            <div style={{ display: 'grid', gap: 12 }}>
              <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
                Transcription layer
                <select
                  aria-label="Transcription layer"
                  value={selectedTranscriptionLayerId ?? ''}
                  onChange={selectTranscriptionLayer}
                >
                  {transcriptionLayers.map((layer) => (
                    <option key={layer.id} value={layer.id}>
                      {layer.name}
                      {layer.kind === 'model' ? ' (read-only)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <Typography.Text>
                {selectedSegmentNumber
                  ? `Selected Segment ${selectedSegmentNumber}`
                  : 'Select a Segment to view transcription text.'}
              </Typography.Text>
              {selectedSegment && selectedTranscriptionLayer?.kind === 'ground_truth' && (
                <>
                  <label style={{ display: 'grid', gap: 8 }}>
                    Ground truth text for selected Segment
                    <Input.TextArea
                      aria-label="Ground truth text for selected Segment"
                      value={approvedTextDraft}
                      rows={3}
                      onChange={(event) => setApprovedTextDraft(event.target.value)}
                    />
                  </label>
                  <Button type="primary" onClick={() => void saveGroundTruthText()}>
                    Save Ground truth text
                  </Button>
                </>
              )}
              {selectedSegment && selectedTranscriptionLayer?.kind === 'model' && (
                <>
                  <label style={{ display: 'grid', gap: 8 }}>
                    Read-only text for selected Segment
                    <Input.TextArea
                      aria-label="Read-only text for selected Segment"
                      value={lineTextForLayer(selectedSegment, selectedTranscriptionLayer.id)}
                      rows={3}
                      readOnly
                    />
                  </label>
                  <Space wrap>
                    <Button
                      type="primary"
                      onClick={() => void copySelectedLayerToGroundTruth([selectedSegment.id])}
                    >
                      Copy selected Segment to Ground truth
                    </Button>
                    <Button onClick={() => void copySelectedLayerToGroundTruth(null)}>
                      Copy whole Page to Ground truth
                    </Button>
                  </Space>
                </>
              )}
              <details>
                <summary>Page transcription and pairing</summary>
                <div style={{ display: 'grid', gap: 8, paddingTop: 8 }}>
                  <Typography.Text>
                    Pairing progress: {pairingProgress.paired_lines}/{pairingProgress.total_lines}{' '}
                    Lines paired
                  </Typography.Text>
                  <label style={{ display: 'grid', gap: 8 }}>
                    Page transcription text
                    <Input.TextArea
                      aria-label="Page transcription text"
                      value={pageTranscriptionText}
                      rows={4}
                      onChange={(event) => setPageTranscriptionText(event.target.value)}
                    />
                  </label>
                  <Button onClick={() => void importPageTranscription()}>
                    Import page transcription
                  </Button>
                </div>
              </details>
              {textLines.map((textLine) => {
                const pairedIndex = textLine.paired_line_id
                  ? [...lines]
                      .sort((a, b) => a.order - b.order)
                      .findIndex((line) => line.id === textLine.paired_line_id)
                  : -1;
                const pairedLabel =
                  pairedIndex >= 0 ? ` · paired with Segment ${pairedIndex + 1}` : '';
                return (
                  <div
                    key={textLine.order}
                    style={{
                      border: '1px solid #3b4350',
                      borderRadius: 6,
                      padding: 8,
                    }}
                  >
                    <Typography.Text style={{ color: '#d8c7a1' }}>
                      Text line {textLine.order + 1}
                      {pairedLabel}
                    </Typography.Text>
                    <Typography.Paragraph style={{ marginBottom: 8 }}>
                      {textLine.text}
                    </Typography.Paragraph>
                    <Button
                      disabled={!selectedSegmentId}
                      onClick={() => void pairTextLine(textLine.order)}
                    >
                      Pair Text line {textLine.order + 1}
                    </Button>
                  </div>
                );
              })}
              {selectedSegmentNumber && (
                <div style={{ display: 'grid', gap: 8 }}>
                  <Typography.Text>
                    Selected Segment {selectedSegmentNumber}
                  </Typography.Text>
                  <label style={{ display: 'grid', gap: 8 }}>
                    Approved text for selected Segment
                    <Input.TextArea
                      aria-label="Approved text for selected Segment"
                      value={approvedTextDraft}
                      rows={3}
                      onChange={(event) => setApprovedTextDraft(event.target.value)}
                    />
                  </label>
                  <Button type="primary" onClick={() => void saveApprovedText()}>
                    Save approved text
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function points(points: LayoutPoint[] | null | undefined): string {
  return (points ?? []).map(([x, y]) => `${x},${y}`).join(' ');
}

function rectanglePoints(start: LinePoint, end: LinePoint): LinePoint[] {
  const [startX, startY] = start;
  const [endX, endY] = end;
  return [
    [Math.min(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.max(startY, endY)],
    [Math.min(startX, endX), Math.max(startY, endY)],
  ];
}

function LayoutCanvas({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  drawingRectangle,
  drawingPolygon,
  onDraftStart,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
}: {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  drawingRectangle: boolean;
  drawingPolygon: boolean;
  onDraftStart: (point: LinePoint) => void;
  onRectangleDrawn: (point: LinePoint) => void;
  onPolygonPoint: (point: LinePoint) => void;
  onPolygonComplete: () => void;
  onSelectLine: (lineId: string) => void;
  onSelectSegment: (lineId: string) => void;
}) {
  const eventPoint = (event: MouseEvent<SVGSVGElement>): LinePoint => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return [event.clientX, event.clientY];
    }
    return [
      Math.round(((event.clientX - rect.left) / rect.width) * imageWidth),
      Math.round(((event.clientY - rect.top) / rect.height) * imageHeight),
    ];
  };

  return (
    <div style={{ position: 'relative', width: 480, maxWidth: '100%' }}>
      <AuthenticatedImage src={imageUrl} alt={imageAlt} width={480} />
      <svg
        role="img"
        aria-label="Page geometry canvas"
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        onMouseDown={(event) => {
          if (drawingRectangle) onDraftStart(eventPoint(event));
        }}
        onMouseUp={(event) => {
          if (drawingRectangle) onRectangleDrawn(eventPoint(event));
        }}
        onClick={(event) => {
          if (drawingPolygon) onPolygonPoint(eventPoint(event));
        }}
        onDoubleClick={() => {
          if (drawingPolygon) onPolygonComplete();
        }}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          cursor: drawingRectangle || drawingPolygon ? 'crosshair' : 'default',
        }}
      >
        {layout.blocks.map((block) => (
          <polygon
            key={block.id}
            aria-label={`Block ${block.id}`}
            points={points(block.box)}
            fill="rgba(216, 199, 161, 0.08)"
            stroke="#d8c7a1"
            strokeWidth={4}
          />
        ))}
        {layout.lines.map((line) => (
          <polyline
            key={line.id}
            aria-label={`Line ${line.id} baseline`}
            onClick={() => onSelectLine(line.id)}
            points={points(line.baseline)}
            fill="none"
            stroke={line.manual_geometry ? '#58d68d' : '#5dade2'}
            strokeLinecap="round"
            strokeWidth={6}
          />
        ))}
        {lines
          .slice()
          .sort((a, b) => a.order - b.order)
          .map((line) => (
            <polygon
              key={line.id}
              aria-label={`Segment ${line.order + 1}`}
              onClick={() => onSelectSegment(line.id)}
              points={points(line.points)}
              fill="rgba(88, 214, 141, 0.14)"
              stroke="#58d68d"
              strokeWidth={5}
            />
          ))}
      </svg>
    </div>
  );
}

