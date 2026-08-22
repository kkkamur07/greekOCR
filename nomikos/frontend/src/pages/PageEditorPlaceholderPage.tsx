import { useCallback, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { type LayoutPoint, type LinePoint } from "../api/client";
import { invalidateAfter } from "../api/resources";
import { useHostPreference } from "../inference";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { PageEditorCanvas } from "../components/page-editor/PageEditorCanvas";
import { PageEditorTranscriptionStrip } from "../components/page-editor/PageEditorTranscriptionStrip";
import { PageEditorShell } from "../components/page-editor/PageEditorShell";
import {
  loadPageEditorSettings,
  savePageEditorSettings,
} from "../components/page-editor/pageEditorSettings";
import {
  PageEditorStatusAlerts,
  hasPageEditorStatusAlerts,
} from "../components/page-editor/PageEditorStatusAlerts";
import { PageEditorInferenceBanner } from "../components/page-editor/PageEditorInferenceBanner";
import { PageEditorToolbar } from "../components/page-editor/PageEditorToolbar";
import { PageEditorTranscriptionPdfWrap } from "../components/page-editor/PageEditorTranscriptionPdfWrap";
import {
  rectanglePoints,
  removePolygonVertex,
} from "../components/page-editor/canvasGeometry";
import {
  useLayoutMutations,
  usePageEditorData,
  usePageEditorJobQueue,
  usePageEditorRunState,
  usePairingState,
} from "../components/page-editor/hooks";
import {
  segmentHasGroundTruth,
  segmentIdsWithGroundTruth,
} from "../components/page-editor/hooks/utils";

export function PageEditorPlaceholderPage() {
  const { projectId, documentId, partId } =
    useParams<{
      projectId: string;
      documentId: string;
      partId: string;
    }>() ?? {};

  const [editorMode, setEditorMode] = useState<"layout" | "transcription">(
    "layout",
  );
  const [drawMode, setDrawMode] = useState<"none" | "rectangle" | "polygon">(
    "none",
  );
  const [draftStart, setDraftStart] = useState<LayoutPoint | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<LinePoint[]>([]);
  const [actionsOpen, setActionsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [canvasSettings, setCanvasSettings] = useState(loadPageEditorSettings);
  const [transcriptionPdfOpen, setTranscriptionPdfOpen] = useState(false);
  const [transcriptionPdfRefreshKey, setTranscriptionPdfRefreshKey] =
    useState(0);
  const [stripDismissed, setStripDismissed] = useState(false);
  const [selectedVertexIndex, setSelectedVertexIndex] = useState<number | null>(
    null,
  );
  const [vertexCommitSignal, setVertexCommitSignal] = useState(0);
  // The platform refused a submission because no **inference host** had
  // capacity. Held on the page, not in a toast: it is an explanation to act on.
  const [submissionRefusal, setSubmissionRefusal] = useState<string | null>(
    null,
  );

  const editorData = usePageEditorData(projectId, documentId, partId, () => {
    setEditorMode("layout");
    setDrawMode("none");
    setDraftPolygon([]);
    setDraftStart(null);
  });
  const {
    document,
    setDocument,
    part,
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

  const jobQueue = usePageEditorJobQueue();
  const hostPreference = useHostPreference();

  /** Turn the account setting off, from the install prompt's escape hatch. */
  function preferCloudInferencePermanently() {
    void hostPreference.setPreferLocalInference(false);
  }

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
    setSubmissionRefusal,
    trackJobAndWait: jobQueue.trackAndWait,
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
    onDrawComplete: () => setDrawMode("none"),
    setSubmissionRefusal,
    trackJobAndWait: jobQueue.trackAndWait,
  });

  const {
    selectedLineId,
    setSelectedLineId,
    setSelectedLineSnapshot,
    saveMessage,
    setSaveMessage,
    mutationError,
    segmenting,
    segmentMessage,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
    replaceWithManualLine,
    updateSegmentPoints,
    deleteSelectedSegment,
    undoEdit,
    redoEdit,
    runAutoSegment,
  } = layoutMutations;

  const {
    selectedSegmentId,
    setSelectedSegmentId,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    ocrRunning,
    ocrScope,
    ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    pairTextLine,
    saveApprovedText,
    selectTranscriptionLayer,
    saveGroundTruthText,
    runSegmentOcr,
    runPageOcr,
    promoteSelectedSegmentToGroundTruth,
    selectSegment,
    navigateSegment,
  } = pairing;

  const pairedIds = useMemo(() => segmentIdsWithGroundTruth(lines), [lines]);
  const stripVisible = Boolean(selectedSegment) && !stripDismissed;

  function handleSelectSegment(lineId: string) {
    setSelectedLineId(null);
    setSaveMessage(null);
    setStripDismissed(false);
    setSelectedVertexIndex(null);
    selectSegment(lineId);
  }

  function handleRemoveSelectedVertex() {
    if (!selectedSegmentId || selectedVertexIndex === null) return;
    const segment = lines.find((line) => line.id === selectedSegmentId);
    if (!segment) return;
    const nextPoints = removePolygonVertex(segment.points, selectedVertexIndex);
    if (!nextPoints) return;
    setSelectedVertexIndex(null);
    void updateSegmentPoints(selectedSegmentId, nextPoints);
  }

  const runState = usePageEditorRunState({ segmenting, ocrRunning, ocrScope });

  const canvasHint = runState.processingLabel
    ? `${runState.processingLabel}…`
    : editorMode === "layout" && drawMode === "polygon"
      ? draftPolygon.length === 0
        ? "Polygon: click to place the first corner"
        : `Polygon: ${draftPolygon.length} point${draftPolygon.length === 1 ? "" : "s"} · click to add · double-click or Enter to finish`
      : editorMode === "layout" && selectedSegment && drawMode === "none"
        ? selectedVertexIndex !== null
          ? `Segment ${selectedSegmentNumber} · vertex ${selectedVertexIndex + 1} selected · Delete removes point · Esc deselects`
          : `Segment ${selectedSegmentNumber} · click edge to add · click handle to select · Delete removes Segment · Esc deselects`
        : selectedSegment
          ? `Segment ${selectedSegmentNumber} selected · ${
              segmentHasGroundTruth(selectedSegment) ? "paired" : "unpaired"
            }`
          : editorMode === "layout"
            ? "Select a segment · click edges/handles to edit shape · Esc to deselect"
            : "Select a segment to view transcription";

  function pickDrawMode(nextMode: "rectangle" | "polygon") {
    setDrawMode((mode) => (mode === nextMode ? "none" : nextMode));
    setDraftPolygon([]);
    setDraftStart(null);
    setActionsOpen(false);
  }

  const handlePanSelect = useCallback(() => {
    setDrawMode("none");
    setDraftPolygon([]);
    setDraftStart(null);
    setActionsOpen(false);
    setVertexCommitSignal((value) => value + 1);
    setSelectedVertexIndex(null);
    setSelectedSegmentId(null);
    setSelectedLineId(null);
    setApprovedTextDraft("");
    setStripDismissed(false);
  }, [setSelectedSegmentId, setSelectedLineId, setApprovedTextDraft]);

  function completeDraftPolygon() {
    if (draftPolygon.length >= 3) {
      void replaceWithManualLine("polygon", draftPolygon);
    }
    setDraftPolygon([]);
  }

  useKeyboardShortcuts({
    onDrawBox:
      editorMode === "layout" ? () => pickDrawMode("rectangle") : undefined,
    onDrawPolygon:
      editorMode === "layout" ? () => pickDrawMode("polygon") : undefined,
    onDelete:
      selectedVertexIndex !== null && selectedSegmentId
        ? () => handleRemoveSelectedVertex()
        : selectedSegmentId || selectedLineId
          ? () => {
              if (selectedSegmentId) void deleteSelectedSegment();
              if (selectedLineId) void resetSelectedLine();
            }
          : undefined,
    onEscape: handlePanSelect,
    onUndo: () => void undoEdit(),
    onRedo: () => void redoEdit(),
    onEnter:
      editorMode === "layout" &&
      drawMode === "polygon" &&
      draftPolygon.length >= 3
        ? completeDraftPolygon
        : undefined,
  });

  function handleCanvasSettingsChange(next: typeof canvasSettings) {
    setCanvasSettings(next);
    savePageEditorSettings(next);
  }

  function openTranscriptionPdf() {
    setTranscriptionPdfRefreshKey(Date.now());
    setTranscriptionPdfOpen(true);
    setActionsOpen(false);
  }

  function refreshTranscriptionPdf() {
    setTranscriptionPdfRefreshKey(Date.now());
  }

  const statusAlertProps = {
    submissionRefusal,
    saveMessage,
    transcriptionSaveMessage,
    ocrMessage,
    segmentMessage,
    mutationError,
    pairingError,
    layoutError,
    lineError,
  };

  const documentHref =
    projectId && documentId
      ? `/projects/${projectId}/documents/${documentId}`
      : "/projects";

  return (
    <PageEditorShell
      loading={loading}
      backHref={documentHref}
      unavailableDescription={
        error || !document || !part
          ? (error ?? "This document part was not found.")
          : null
      }
      showStatusAlerts={hasPageEditorStatusAlerts(statusAlertProps)}
      statusAlerts={<PageEditorStatusAlerts {...statusAlertProps} />}
      inferenceBanner={
        <PageEditorInferenceBanner
          hasLocalCapacity={hostPreference.hasLocalCapacity}
          loading={hostPreference.loading}
          preferLocalInference={hostPreference.preferLocalInference}
          onRetry={() => void hostPreference.refresh()}
          onUseCloudInstead={preferCloudInferencePermanently}
        />
      }
      toolbar={
        document && part ? (
          <PageEditorToolbar
            projectId={projectId}
            documentId={documentId}
            partId={part.id}
            document={document}
            partIndex={partIndex ?? 1}
            editorMode={editorMode}
            onEditorModeChange={(mode) => {
              setEditorMode(mode);
              setDrawMode("none");
              setActionsOpen(false);
            }}
            drawMode={drawMode}
            onPickDrawMode={pickDrawMode}
            onPanSelect={handlePanSelect}
            lines={lines}
            pairingProgress={pairingProgress}
            selectedSegmentId={selectedSegmentId}
            selectedLineId={selectedLineId}
            textLines={textLines}
            onPairTextLine={pairTextLine}
            onDocumentWorkflowChange={(workflow) => {
              setDocument((current) =>
                current ? { ...current, workflow } : current,
              );
              // Publishing is the one write in this editor that changes what a
              // reader can reach at all. This handler used to stop at the local
              // copy above, so the document list, the detail page and the
              // public page went on showing the old status.
              if (projectId && documentId) {
                invalidateAfter.documentUpdated(projectId, documentId);
              }
            }}
            onDeleteSelectedSegment={deleteSelectedSegment}
            onResetSelectedLine={resetSelectedLine}
            actionsOpen={actionsOpen}
            onActionsOpenChange={setActionsOpen}
            segmenting={segmenting}
            ocrRunning={ocrRunning}
            ocrScope={ocrScope}
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={setSelectedTranscribeModelId}
            onRunAutoSegment={runAutoSegment}
            onRunSegmentOcr={runSegmentOcr}
            onRunPageOcr={runPageOcr}
            transcriptionPdfOpen={transcriptionPdfOpen}
            onOpenTranscriptionPdf={openTranscriptionPdf}
            onCloseTranscriptionPdf={() => setTranscriptionPdfOpen(false)}
            settingsOpen={settingsOpen}
            onSettingsOpenChange={setSettingsOpen}
            canvasSettings={canvasSettings}
            onCanvasSettingsChange={handleCanvasSettingsChange}
            preferLocalInference={hostPreference.preferLocalInference}
            onPreferLocalInferenceChange={(preferLocal) =>
              void hostPreference.setPreferLocalInference(preferLocal)
            }
            preferenceSaving={hostPreference.saving}
            hasLocalCapacity={hostPreference.hasLocalCapacity}
            hostPreferenceLoading={hostPreference.loading}
          />
        ) : null
      }
    >
      {document && part && (
        <div className="pe-workspace">
          <div className="pe-body">
            <div className="pe-canvas-pane">
              <PageEditorCanvas
                imageUrl={part.image_url}
                imageAlt={`Page ${partIndex}`}
                imageWidth={part.width ?? 640}
                imageHeight={part.height ?? 900}
                layout={layout}
                lines={lines}
                selectedSegmentId={selectedSegmentId}
                pairedSegmentIds={pairedIds}
                settings={canvasSettings}
                drawingRectangle={drawMode === "rectangle"}
                drawingPolygon={drawMode === "polygon"}
                draftStart={draftStart}
                draftPolygon={draftPolygon}
                onDraftStart={setDraftStart}
                onRectangleDrawn={async (end) => {
                  if (!draftStart) return;
                  const rectangle = rectanglePoints(draftStart, end);
                  await replaceWithManualLine("rectangle", rectangle);
                  setDraftStart(null);
                }}
                onPolygonPoint={(point) =>
                  setDraftPolygon((current) => [...current, point])
                }
                onPolygonComplete={completeDraftPolygon}
                onSelectLine={(lineId) => {
                  const selectedLine = layout.lines.find(
                    (line) => line.id === lineId,
                  );
                  setSelectedLineId(lineId);
                  setSelectedSegmentId(null);
                  setSelectedVertexIndex(null);
                  setSelectedLineSnapshot({
                    baseline: selectedLine?.baseline,
                    mask: selectedLine?.mask,
                  });
                }}
                onSelectSegment={handleSelectSegment}
                segmentVertexEditEnabled={
                  editorMode === "layout" &&
                  drawMode === "none" &&
                  Boolean(selectedSegmentId)
                }
                selectedVertexIndex={selectedVertexIndex}
                onSelectedVertexChange={setSelectedVertexIndex}
                commitSignal={vertexCommitSignal}
                onSegmentPointsChange={updateSegmentPoints}
              />
              <p
                className={`pe-canvas-hint${runState.processingKind ? " pe-canvas-hint--processing" : ""}`}
                id="canvas-hint"
                role="status"
              >
                {canvasHint}
              </p>
              <div className="pe-seg-legend" aria-label="Segment pairing">
                <div className="pe-seg-legend__item">
                  <span className="pe-seg-legend__swatch pe-seg-legend__swatch--paired" />
                  paired
                </div>
                <div className="pe-seg-legend__item">
                  <span className="pe-seg-legend__swatch pe-seg-legend__swatch--unpaired" />
                  unpaired
                </div>
              </div>
            </div>
            {transcriptionPdfOpen && projectId && documentId && partId && (
              <PageEditorTranscriptionPdfWrap
                projectId={projectId}
                documentId={documentId}
                partId={partId}
                downloadFilename={`${document.name.replace(/\s+/g, "_")}_page_${partIndex}_transcription.pdf`}
                refreshKey={transcriptionPdfRefreshKey}
                onClose={() => setTranscriptionPdfOpen(false)}
                onRefresh={refreshTranscriptionPdf}
              />
            )}
          </div>

          {selectedLineId && (
            <div className="pe-baseline-bar">
              <span>Selected baseline</span>
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => moveSelectedBaseline(5)}
              >
                Move baseline down
              </button>
              <button
                type="button"
                className="btn btn--active btn--sm"
                onClick={() => void saveSelectedLine()}
              >
                Save layout
              </button>
              <button
                type="button"
                className="btn btn--danger-ghost btn--sm"
                onClick={() => void resetSelectedLine()}
              >
                Reset layout
              </button>
            </div>
          )}

          <PageEditorTranscriptionStrip
            visible={stripVisible}
            transcriptionLayers={transcriptionLayers}
            selectedTranscriptionLayerId={selectedTranscriptionLayerId}
            onSelectTranscriptionLayer={selectTranscriptionLayer}
            selectedSegmentNumber={selectedSegmentNumber}
            selectedSegment={selectedSegment}
            selectedTranscriptionLayer={selectedTranscriptionLayer}
            approvedTextDraft={approvedTextDraft}
            onApprovedTextDraftChange={setApprovedTextDraft}
            onSaveGroundTruthText={async () => {
              if (await saveGroundTruthText()) setStripDismissed(true);
            }}
            onPromoteSelectedSegmentToGroundTruth={
              promoteSelectedSegmentToGroundTruth
            }
            onRunSegmentOcr={runSegmentOcr}
            onNavigateSegment={navigateSegment}
            onDismiss={() => setStripDismissed(true)}
            lines={lines}
            selectedSegmentId={selectedSegmentId}
            onSaveApprovedText={async () => {
              if (await saveApprovedText()) setStripDismissed(true);
            }}
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={setSelectedTranscribeModelId}
            ocrRunning={ocrRunning}
            ocrScope={ocrScope}
            backgroundJobsActive={runState.backgroundJobsActive}
          />
        </div>
      )}
    </PageEditorShell>
  );
}
