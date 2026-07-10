import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { type LayoutPoint, type LinePoint, api } from "../api/client";
import {
  DEFAULT_SEGMENT_REGISTRY_MODEL_ID,
  fetchLocalCacheStatus,
  isModelRemoteOnly,
  registrySelectionFromArtifactRef,
  useInferenceHost,
} from "../inference";
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
import { PageEditorLocalInferenceBanner } from "../components/page-editor/PageEditorLocalInferenceBanner";
import { PageEditorToolbar } from "../components/page-editor/PageEditorToolbar";
import {
  getPageEditorProcessingLabel,
  type PageEditorProcessingKind,
} from "../components/page-editor/PageEditorProcessingBanner";
import { PageEditorTranscriptionPdfWrap } from "../components/page-editor/PageEditorTranscriptionPdfWrap";
import { rectanglePoints } from "../components/page-editor/canvasGeometry";
import {
  useLayoutMutations,
  usePageEditorData,
  usePageEditorJobQueue,
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
  const inferenceHost = useInferenceHost();
  const [segmentRegistryModelId, setSegmentRegistryModelId] = useState<
    string | null
  >(null);
  const [localInferenceModelId, setLocalInferenceModelId] = useState<
    string | null
  >(null);
  const localInferenceAbortRef = useRef<AbortController | null>(null);
  const switchToCloudRef = useRef(false);

  const localInference = useMemo(
    () => ({
      onStart: async (registryModelId: string, registryTag = "stable") => {
        localInferenceAbortRef.current?.abort();
        localInferenceAbortRef.current = new AbortController();
        // Only surface the download banner the first time a model is used on this
        // machine. Once the weights are cached locally, the run proceeds silently.
        const cached = await fetchLocalCacheStatus(
          registryModelId,
          registryTag,
        );
        setLocalInferenceModelId(cached === false ? registryModelId : null);
      },
      onEnd: () => {
        setLocalInferenceModelId(null);
        localInferenceAbortRef.current = null;
      },
      getSignal: () => localInferenceAbortRef.current?.signal,
      shouldFallbackToCloud: () => switchToCloudRef.current,
      clearFallbackToCloud: () => {
        switchToCloudRef.current = false;
      },
    }),
    [],
  );

  function useCloudInstead() {
    switchToCloudRef.current = true;
    localInferenceAbortRef.current?.abort();
    inferenceHost.setInferencePreference("cloud");
    setLocalInferenceModelId(null);
  }

  useEffect(() => {
    if (!projectId || !documentId || !partId) {
      setSegmentRegistryModelId(null);
      return;
    }
    let cancelled = false;
    void api
      .resolvePartModelBinding(projectId, documentId, partId, "segment")
      .then((resolved) => {
        if (cancelled) return;
        const { registryModelId } = registrySelectionFromArtifactRef(
          resolved.model.artifact_ref,
        );
        setSegmentRegistryModelId(registryModelId);
      })
      .catch(() => {
        if (!cancelled) {
          setSegmentRegistryModelId(DEFAULT_SEGMENT_REGISTRY_MODEL_ID);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, documentId, partId]);

  const partImageUrl = part?.image_url ?? null;
  const shouldUseLocalPath = useCallback(
    (registryModelId: string) =>
      inferenceHost.shouldUseLocalPath(registryModelId),
    [inferenceHost],
  );

  const selectedModelHostEligibility = useMemo(() => {
    const model = transcribeModels.find(
      (entry) => entry.id === selectedTranscribeModelId,
    );
    if (!model) return null;
    try {
      const { registryModelId } = registrySelectionFromArtifactRef(
        model.artifact_ref,
      );
      if (isModelRemoteOnly(inferenceHost.catalog, registryModelId)) {
        return "remote" as const;
      }
      if (inferenceHost.shouldUseLocalPath(registryModelId)) {
        return "local" as const;
      }
      return "any" as const;
    } catch {
      return null;
    }
  }, [inferenceHost, selectedTranscribeModelId, transcribeModels]);

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
    transcribeModels,
    partImageUrl,
    shouldUseLocalPath,
    localInference,
    trackJobAndWait: jobQueue.trackAndWait,
    trackLocalTask: jobQueue.trackLocalTask,
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
    partImageUrl,
    shouldUseLocalPath,
    segmentRegistryModelId,
    localInference,
    trackJobAndWait: jobQueue.trackAndWait,
    trackLocalTask: jobQueue.trackLocalTask,
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
    otsuSphereRadius,
    setOtsuSphereRadius,
    segmentMessage,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
    replaceWithManualLine,
    updateSegmentPoints,
    deleteSelectedSegment,
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
    selectSegment(lineId);
  }

  const processingKind: PageEditorProcessingKind = segmenting
    ? "segmentation"
    : ocrRunning
      ? ocrScope === "page"
        ? "transcription-page"
        : "transcription-segment"
      : null;

  const canvasHint = processingKind
    ? `${getPageEditorProcessingLabel(processingKind)}…`
    : editorMode === "layout" && drawMode === "polygon"
      ? draftPolygon.length === 0
        ? "Polygon: click to place the first corner"
        : `Polygon: ${draftPolygon.length} point${draftPolygon.length === 1 ? "" : "s"} · click to add · double-click or Enter to finish`
      : editorMode === "layout" && selectedSegment && drawMode === "none"
        ? `Segment ${selectedSegmentNumber} · click edge to add · click handle to remove · drag to move`
        : selectedSegment
          ? `Segment ${selectedSegmentNumber} selected · ${
              segmentHasGroundTruth(selectedSegment) ? "paired" : "unpaired"
            }`
          : editorMode === "layout"
            ? "Select a segment · click edges/handles to edit shape"
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
  }, []);

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
      selectedSegmentId || selectedLineId
        ? () => {
            if (selectedSegmentId) void deleteSelectedSegment();
            if (selectedLineId) void resetSelectedLine();
          }
        : undefined,
    onEscape: handlePanSelect,
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
      processingBanner={null}
      inferenceBanner={
        <>
          <PageEditorLocalInferenceBanner
            registryModelId={localInferenceModelId}
            onUseCloudInstead={useCloudInstead}
          />
          <PageEditorInferenceBanner
            helperAvailable={inferenceHost.helperAvailable}
            probing={inferenceHost.probing}
            preferCloud={inferenceHost.preferCloud}
            onUseCloudInstead={useCloudInstead}
          />
        </>
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
            onDocumentWorkflowChange={(workflow) =>
              setDocument((current) =>
                current ? { ...current, workflow } : current,
              )
            }
            onDeleteSelectedSegment={deleteSelectedSegment}
            onResetSelectedLine={resetSelectedLine}
            actionsOpen={actionsOpen}
            onActionsOpenChange={setActionsOpen}
            useOtsuRefinement={useOtsuRefinement}
            onUseOtsuRefinementChange={setUseOtsuRefinement}
            otsuSphereRadius={otsuSphereRadius}
            onOtsuSphereRadiusChange={setOtsuSphereRadius}
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
            inferencePreference={inferenceHost.preference}
            onInferencePreferenceChange={inferenceHost.setInferencePreference}
            helperAvailable={inferenceHost.helperAvailable}
            helperProbing={inferenceHost.probing}
            preferCloud={inferenceHost.preferCloud}
            selectedModelHostEligibility={selectedModelHostEligibility}
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
                onSegmentPointsChange={updateSegmentPoints}
              />
              <p
                className={`pe-canvas-hint${processingKind ? " pe-canvas-hint--processing" : ""}`}
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
            onSaveGroundTruthText={saveGroundTruthText}
            onPromoteSelectedSegmentToGroundTruth={
              promoteSelectedSegmentToGroundTruth
            }
            onRunSegmentOcr={runSegmentOcr}
            onNavigateSegment={navigateSegment}
            onDismiss={() => setStripDismissed(true)}
            lines={lines}
            selectedSegmentId={selectedSegmentId}
            onSaveApprovedText={saveApprovedText}
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={setSelectedTranscribeModelId}
            ocrRunning={ocrRunning}
            ocrScope={ocrScope}
            backgroundJobsActive={jobQueue.activeCount > 0}
          />
        </div>
      )}
    </PageEditorShell>
  );
}
