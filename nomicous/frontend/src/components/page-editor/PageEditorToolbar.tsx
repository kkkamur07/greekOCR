import { useEffect, useRef } from "react";
import Link from "next/link";
import type {
  DocumentWithPartsResponse,
  InferenceModelResponse,
  LineResponse,
} from "../../api/client";
import { PageEditorBackLink } from "./PageEditorNavHeader";
import { editorButton } from "./editorButton";
import { PageEditorModelSelect } from "./PageEditorModelSelect";
import { PageEditorSharingMenu } from "./PageEditorSharingMenu";
import { SettingsIcon } from "./EditorIcons";
import { PageEditorSettingsPanel } from "./PageEditorSettingsPanel";
import { PageEditorInferenceStatus } from "./PageEditorInferenceStatus";
import { PAGE_EDITOR_SHORTCUTS } from "./pageEditorShortcuts";
import type { InferencePreference } from "../../inference/preference";
import type { HostEligibility } from "../../inference/types";
import type { PageEditorCanvasSettings } from "./pageEditorSettings";
import { ToolbarKbd } from "./ToolbarKbd";

type PageEditorToolbarProps = {
  projectId: string | undefined;
  documentId: string | undefined;
  document: DocumentWithPartsResponse;
  partIndex: number;
  editorMode: "layout" | "transcription";
  onEditorModeChange: (mode: "layout" | "transcription") => void;
  drawMode: "none" | "rectangle" | "polygon";
  onPickDrawMode: (mode: "rectangle" | "polygon") => void;
  onPanSelect: () => void;
  lines: LineResponse[];
  pairingProgress: {
    paired_lines: number;
    total_lines: number;
    percent: number;
  };
  partId: string;
  selectedSegmentId: string | null;
  selectedLineId: string | null;
  textLines: { order: number; text: string; paired_line_id: string | null }[];
  onPairTextLine: (order: number) => void;
  onDocumentWorkflowChange: (
    workflow: DocumentWithPartsResponse["workflow"],
  ) => void;
  onDeleteSelectedSegment: () => void;
  onResetSelectedLine: () => void;
  actionsOpen: boolean;
  onActionsOpenChange: (open: boolean) => void;
  useOtsuRefinement: boolean;
  onUseOtsuRefinementChange: (value: boolean) => void;
  otsuSphereRadius: number;
  onOtsuSphereRadiusChange: (value: number) => void;
  segmenting: boolean;
  ocrRunning: boolean;
  ocrScope?: "segment" | "page" | null;
  transcribeModels: InferenceModelResponse[];
  selectedTranscribeModelId: string | null;
  onSelectedTranscribeModelIdChange: (modelId: string | null) => void;
  onRunAutoSegment: () => void;
  onRunSegmentOcr: () => void;
  onRunPageOcr: () => void;
  transcriptionPdfOpen: boolean;
  onOpenTranscriptionPdf: () => void;
  onCloseTranscriptionPdf: () => void;
  settingsOpen: boolean;
  onSettingsOpenChange: (open: boolean) => void;
  canvasSettings: PageEditorCanvasSettings;
  onCanvasSettingsChange: (settings: PageEditorCanvasSettings) => void;
  inferencePreference: InferencePreference;
  onInferencePreferenceChange: (preference: InferencePreference) => void;
  helperAvailable: boolean;
  helperProbing: boolean;
  preferCloud: boolean;
  selectedModelHostEligibility: HostEligibility | null;
};

export function PageEditorToolbar({
  projectId,
  documentId,
  document,
  partIndex,
  editorMode,
  onEditorModeChange,
  drawMode,
  onPickDrawMode,
  onPanSelect,
  lines,
  pairingProgress,
  selectedSegmentId,
  selectedLineId,
  textLines,
  onPairTextLine,
  onDocumentWorkflowChange,
  onDeleteSelectedSegment,
  onResetSelectedLine,
  actionsOpen,
  onActionsOpenChange,
  useOtsuRefinement,
  onUseOtsuRefinementChange,
  otsuSphereRadius,
  onOtsuSphereRadiusChange,
  segmenting,
  ocrRunning,
  ocrScope = null,
  transcribeModels,
  selectedTranscribeModelId,
  onSelectedTranscribeModelIdChange,
  onRunAutoSegment,
  onRunSegmentOcr,
  onRunPageOcr,
  transcriptionPdfOpen,
  onOpenTranscriptionPdf,
  onCloseTranscriptionPdf,
  settingsOpen,
  onSettingsOpenChange,
  canvasSettings,
  onCanvasSettingsChange,
  inferencePreference,
  onInferencePreferenceChange,
  helperAvailable,
  helperProbing,
  preferCloud,
  selectedModelHostEligibility,
}: PageEditorToolbarProps) {
  const dropdownRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!actionsOpen && !settingsOpen) return;
    function handleClick(event: MouseEvent) {
      const target = event.target as Node;
      if (
        actionsOpen &&
        dropdownRef.current &&
        !dropdownRef.current.contains(target)
      ) {
        onActionsOpenChange(false);
      }
      if (
        settingsOpen &&
        settingsRef.current &&
        !settingsRef.current.contains(target)
      ) {
        onSettingsOpenChange(false);
      }
    }
    globalThis.document.addEventListener("click", handleClick);
    return () => globalThis.document.removeEventListener("click", handleClick);
  }, [actionsOpen, settingsOpen, onActionsOpenChange, onSettingsOpenChange]);

  const segmentLabel = lines.length === 1 ? "seg" : "segs";
  const pairingPercent =
    pairingProgress.total_lines > 0
      ? Math.round(
          (pairingProgress.paired_lines / pairingProgress.total_lines) * 100,
        )
      : 0;
  const processing = segmenting || ocrRunning;
  const processingLabel = segmenting
    ? "Segmenting"
    : ocrRunning
      ? ocrScope === "page"
        ? "Transcribing page"
        : "Transcribing"
      : null;

  return (
    <header className="pe-toolbar" role="banner">
      <span className="visually-hidden">ANNOTE PAGE WORKSPACE</span>
      <h2 className="visually-hidden">
        {editorMode === "layout" ? "Layout edit" : "Transcription edit"}
      </h2>
      <span className="visually-hidden">
        Pairing progress: {pairingProgress.paired_lines}/
        {pairingProgress.total_lines} Lines paired
      </span>
      <span className="visually-hidden">
        {lines.length} {lines.length === 1 ? "Segment" : "Segments"}
      </span>

      <div className="visually-hidden">
        {selectedSegmentId &&
          textLines.map((textLine) => (
            <button
              key={textLine.order}
              type="button"
              disabled={!selectedSegmentId}
              onClick={() => void onPairTextLine(textLine.order)}
            >
              Pair Text line {textLine.order + 1}
            </button>
          ))}
      </div>

      <span className="visually-hidden">
        {document.name} · Page {partIndex}
      </span>

      <Link
        href="/projects"
        className="pe-toolbar__logo"
        aria-label="nomicous home"
      >
        <img src="/nomos.svg" alt="" />
        <span>nomicous</span>
      </Link>

      <div className="pe-toolbar__title">
        {projectId && documentId && (
          <PageEditorBackLink
            to={`/projects/${projectId}/documents/${documentId}`}
          />
        )}
        <div className="pe-toolbar__sep" aria-hidden="true" />
        <h1
          className="pe-toolbar__doc"
          title={`${document.name} · Page ${partIndex}`}
        >
          {document.name}
          <span className="pe-toolbar__doc-page"> · p.{partIndex}</span>
        </h1>
      </div>

      <div className="pe-toolbar__center" aria-label="Page statistics">
        <PageEditorInferenceStatus
          probing={helperProbing}
          helperAvailable={helperAvailable}
          preferCloud={preferCloud}
        />
        {processingLabel && (
          <div
            className="pe-toolbar__processing"
            role="status"
            aria-live="polite"
          >
            <span className="pe-toolbar__processing-dot" aria-hidden="true" />
            {processingLabel}
          </div>
        )}
        <div className="pe-toolbar__stat">
          <strong>{lines.length}</strong> {segmentLabel}
        </div>
        <div
          className="pe-toolbar__progress"
          title={`${pairingProgress.paired_lines} of ${pairingProgress.total_lines} lines paired`}
        >
          <span className="pe-toolbar__stat">
            <strong>{pairingProgress.paired_lines}</strong>/
            {pairingProgress.total_lines}
          </span>
          <div
            className="pe-toolbar__progress-track"
            role="progressbar"
            aria-valuenow={pairingPercent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Pairing progress"
          >
            <div
              className="pe-toolbar__progress-fill"
              style={{ width: `${pairingPercent}%` }}
            />
          </div>
        </div>
      </div>

      <div className="pe-toolbar__actions">
        <div
          className="pe-toolbar__modes"
          role="group"
          aria-label="Editor mode"
        >
          <button
            type="button"
            className={`pe-toolbar__mode ${editorMode === "layout" ? "pe-toolbar__mode--active" : ""}`}
            aria-pressed={editorMode === "layout"}
            onClick={() => onEditorModeChange("layout")}
          >
            Layout
          </button>
          <button
            type="button"
            className={`pe-toolbar__mode ${editorMode === "transcription" ? "pe-toolbar__mode--active" : ""}`}
            aria-label="Transcription edit"
            aria-pressed={editorMode === "transcription"}
            onClick={() => onEditorModeChange("transcription")}
          >
            Transcription
          </button>
        </div>

        <div
          className="pe-toolbar__cluster"
          role="group"
          aria-label="Drawing tools"
        >
          <button
            type="button"
            onClick={onPanSelect}
            className={editorButton(
              drawMode === "none" && editorMode === "layout",
            )}
            title={`Select / pan (${PAGE_EDITOR_SHORTCUTS.SELECT})`}
            aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.SELECT}
          >
            Select
          </button>
          <button
            type="button"
            aria-label={`Rectangle segment (${PAGE_EDITOR_SHORTCUTS.RECTANGLE})`}
            aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.RECTANGLE}
            title={`Draw rectangle segment (${PAGE_EDITOR_SHORTCUTS.RECTANGLE})`}
            onClick={() => onPickDrawMode("rectangle")}
            className={editorButton(drawMode === "rectangle")}
          >
            Rect
            <ToolbarKbd>{PAGE_EDITOR_SHORTCUTS.RECTANGLE}</ToolbarKbd>
          </button>
          <button
            type="button"
            aria-label={`Polygon segment (${PAGE_EDITOR_SHORTCUTS.POLYGON})`}
            aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.POLYGON}
            title={`Draw polygon segment (${PAGE_EDITOR_SHORTCUTS.POLYGON})`}
            onClick={() => onPickDrawMode("polygon")}
            className={editorButton(drawMode === "polygon")}
          >
            Poly
            <ToolbarKbd>{PAGE_EDITOR_SHORTCUTS.POLYGON}</ToolbarKbd>
          </button>
          <button
            type="button"
            aria-label={`Delete segment (${PAGE_EDITOR_SHORTCUTS.DELETE})`}
            aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.DELETE}
            disabled={!selectedSegmentId && !selectedLineId}
            onClick={() => {
              if (selectedSegmentId) void onDeleteSelectedSegment();
              if (selectedLineId) void onResetSelectedLine();
            }}
            className="pe-tb-btn"
            title={`Delete selected (${PAGE_EDITOR_SHORTCUTS.DELETE})`}
          >
            Del
            <ToolbarKbd>{PAGE_EDITOR_SHORTCUTS.DELETE}</ToolbarKbd>
          </button>
        </div>

        <div className="pe-toolbar__cluster">
          <PageEditorModelSelect
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={
              onSelectedTranscribeModelIdChange
            }
            disabled={processing}
          />
        </div>

        <div className="pe-toolbar__cluster pe-dropdown-wrap" ref={dropdownRef}>
          <button
            type="button"
            aria-haspopup="menu"
            aria-expanded={actionsOpen}
            onClick={() => onActionsOpenChange(!actionsOpen)}
            className={`pe-tb-btn${actionsOpen ? " pe-tb-btn--on" : ""}`}
          >
            Process ▾
          </button>
          {actionsOpen && (
            <div
              className="pe-dropdown"
              role="menu"
              aria-label="Processing actions"
            >
              <div className="pe-dd-section">Segment</div>
              <p className="pe-dd-model">
                Engine <strong>blla-segment</strong> (fixed)
              </p>
              <label className="pe-dd-check">
                <input
                  type="checkbox"
                  checked={useOtsuRefinement}
                  disabled={processing}
                  aria-label="Refine Kraken segments with Otsu"
                  onChange={(event) =>
                    onUseOtsuRefinementChange(event.target.checked)
                  }
                  onClick={(event) => event.stopPropagation()}
                />
                Otsu refinement
              </label>
              <div className="pe-dd-field">
                <label htmlFor="otsu-sphere-px">Sphere (px)</label>
                <input
                  id="otsu-sphere-px"
                  type="number"
                  className="pe-dd-field__input"
                  aria-label="Otsu morphological sphere radius in pixels"
                  min={1}
                  step={1}
                  value={otsuSphereRadius}
                  disabled={!useOtsuRefinement || processing}
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => {
                    const next = event.target.valueAsNumber;
                    if (Number.isFinite(next) && next > 0) {
                      onOtsuSphereRadiusChange(next);
                    }
                  }}
                  onBlur={(event) => {
                    const next = event.target.valueAsNumber;
                    if (!Number.isFinite(next) || next <= 0) {
                      onOtsuSphereRadiusChange(4);
                    }
                  }}
                />
              </div>
              <button
                type="button"
                role="menuitem"
                disabled={processing}
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunAutoSegment();
                }}
                className="pe-dd-item"
              >
                {segmenting ? "Segmenting…" : "Auto segment page"}
              </button>
              <div className="pe-dd-divider" />
              <div className="pe-dd-section">HTR</div>
              <p className="pe-dd-model">
                Model{" "}
                <strong>
                  {transcribeModels.find(
                    (m) => m.id === selectedTranscribeModelId,
                  )?.name ?? "not selected"}
                </strong>
              </p>
              <button
                type="button"
                role="menuitem"
                disabled={
                  !selectedSegmentId || processing || !selectedTranscribeModelId
                }
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunSegmentOcr();
                }}
                className="pe-dd-item"
              >
                {ocrRunning ? "OCR…" : "OCR selected segment"}
              </button>
              <button
                type="button"
                role="menuitem"
                disabled={
                  processing || lines.length === 0 || !selectedTranscribeModelId
                }
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunPageOcr();
                }}
                className="pe-dd-item"
              >
                {ocrRunning ? "OCR…" : "OCR full page"}
              </button>
              {projectId && documentId && (
                <PageEditorSharingMenu
                  projectId={projectId}
                  documentId={documentId}
                  workflow={document.workflow}
                  onWorkflowChange={onDocumentWorkflowChange}
                  disabled={processing}
                />
              )}
            </div>
          )}
        </div>

        <div className="pe-toolbar__cluster">
          <button
            type="button"
            className={`pe-tb-btn${transcriptionPdfOpen ? " pe-tb-btn--on" : ""}`}
            aria-pressed={transcriptionPdfOpen}
            aria-label="Toggle transcription PDF"
            title="Toggle transcription PDF"
            onClick={() => {
              if (transcriptionPdfOpen) onCloseTranscriptionPdf();
              else onOpenTranscriptionPdf();
            }}
          >
            PDF
          </button>
          <div className="pe-dropdown-wrap" ref={settingsRef}>
            <button
              type="button"
              className={`pe-tb-btn pe-tb-btn--icon${settingsOpen ? " pe-tb-btn--on" : ""}`}
              aria-haspopup="dialog"
              aria-expanded={settingsOpen}
              aria-label="Editor settings"
              title="Editor settings"
              onClick={() => onSettingsOpenChange(!settingsOpen)}
            >
              <SettingsIcon />
            </button>
            {settingsOpen && (
              <PageEditorSettingsPanel
                settings={canvasSettings}
                onSettingsChange={onCanvasSettingsChange}
                inferencePreference={inferencePreference}
                onInferencePreferenceChange={onInferencePreferenceChange}
                helperAvailable={helperAvailable}
                selectedModelHostEligibility={selectedModelHostEligibility}
              />
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
