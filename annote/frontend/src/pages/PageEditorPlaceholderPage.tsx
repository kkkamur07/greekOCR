import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Button, Space, Typography } from 'antd';
import { api, type LayoutPoint, type LinePoint } from '../api/client';
import { ApiError } from '../api/errors';
import { PageEditorCanvas } from '../components/page-editor/PageEditorCanvas';
import { PageEditorPairingStrip } from '../components/page-editor/PageEditorPairingStrip';
import { PageEditorShell } from '../components/page-editor/PageEditorShell';
import {
  PageEditorStatusAlerts,
  hasPageEditorStatusAlerts,
} from '../components/page-editor/PageEditorStatusAlerts';
import { PageEditorToolbar } from '../components/page-editor/PageEditorToolbar';
import { rectanglePoints } from '../components/page-editor/canvasGeometry';
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

  const statusAlertProps = {
    saveMessage,
    transcriptionSaveMessage,
    copyMessage,
    ocrMessage,
    segmentMessage,
    mutationError,
    pairingError,
    reviewError,
    layoutError,
    lineError,
  };

  return (
    <PageEditorShell
      loading={loading}
      unavailableDescription={
        error || !document || !part
          ? (error ?? 'This document part was not found.')
          : null
      }
      showStatusAlerts={hasPageEditorStatusAlerts(statusAlertProps)}
      statusAlerts={<PageEditorStatusAlerts {...statusAlertProps} />}
      toolbar={
        document && part ? (
          <PageEditorToolbar
            projectId={projectId}
            documentId={documentId}
            document={document}
            part={part}
            partIndex={partIndex}
            editorMode={editorMode}
            onEditorModeChange={(mode) => {
              setEditorMode(mode);
              setDrawMode('none');
              setActionsOpen(false);
            }}
            drawMode={drawMode}
            onPickDrawMode={pickDrawMode}
            onPanSelect={() => {
              setDrawMode('none');
              setActionsOpen(false);
            }}
            lines={lines}
            pairingProgress={pairingProgress}
            selectedSegmentId={selectedSegmentId}
            selectedSegment={selectedSegment}
            selectedLineId={selectedLineId}
            pageTranscriptionText={pageTranscriptionText}
            onPageTranscriptionTextChange={setPageTranscriptionText}
            onImportPageTranscription={importPageTranscription}
            textLines={textLines}
            onPairTextLine={pairTextLine}
            onMoveSelectedSegmentRight={moveSelectedSegmentRight}
            onDeleteSelectedSegment={deleteSelectedSegment}
            onResetSelectedLine={resetSelectedLine}
            actionsOpen={actionsOpen}
            onActionsOpenChange={setActionsOpen}
            useOtsuRefinement={useOtsuRefinement}
            onUseOtsuRefinementChange={setUseOtsuRefinement}
            segmenting={segmenting}
            ocrRunning={ocrRunning}
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={setSelectedTranscribeModelId}
            onRunAutoSegment={runAutoSegment}
            onRunSegmentOcr={runSegmentOcr}
            onRunPageOcr={runPageOcr}
            onUpdateReviewStatus={updateReviewStatus}
          />
        ) : null
      }
    >
      {document && part && (
        <>
          <div
          style={{
            position: 'relative',
            minHeight: 0,
            flex: 1,
            overflow: 'hidden',
            background: '#f5f5f5',
          }}
        >
          <PageEditorCanvas
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

        <PageEditorPairingStrip
          visible={Boolean(selectedSegment) || editorMode === 'transcription'}
          transcriptionLayers={transcriptionLayers}
          selectedTranscriptionLayerId={selectedTranscriptionLayerId}
          onSelectTranscriptionLayer={selectTranscriptionLayer}
          selectedSegmentNumber={selectedSegmentNumber}
          selectedSegment={selectedSegment}
          selectedTranscriptionLayer={selectedTranscriptionLayer}
          approvedTextDraft={approvedTextDraft}
          onApprovedTextDraftChange={setApprovedTextDraft}
          lineTextForLayer={lineTextForLayer}
          onSaveGroundTruthText={saveGroundTruthText}
          onCopySelectedLayerToGroundTruth={copySelectedLayerToGroundTruth}
          pairingProgress={pairingProgress}
          pageTranscriptionText={pageTranscriptionText}
          onPageTranscriptionTextChange={setPageTranscriptionText}
          onImportPageTranscription={importPageTranscription}
          textLines={textLines}
          lines={lines}
          selectedSegmentId={selectedSegmentId}
          onPairTextLine={pairTextLine}
          onSaveApprovedText={saveApprovedText}
        />
        </>
      )}
    </PageEditorShell>
  );
}
