"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ImageCanvas, { type ImageCanvasHandle } from "@/components/ImageCanvas/ImageCanvas";
import { MIN_SEGMENT_POINTS } from "@/components/ImageCanvas/SegmentOverlay";
import EditorHeader from "@/components/editor/EditorHeader";
import ExportPreviewModal from "@/components/ExportPreviewModal";
import SplitPaneDivider from "@/components/SplitPaneDivider";
import HistoryPanel from "@/components/HistoryPanel";
import SegmentPairingBar from "@/components/SegmentPairingBar";
import TranscriptionPdfPanel, { type TranscriptionPdfMode } from "@/components/TranscriptionPdfPanel";
import { useHorizontalSplit } from "@/hooks/useHorizontalSplit";
import { useEditorShortcuts } from "@/hooks/useEditorShortcuts";
import {
  autoSegmentPage,
  binarizePage,
  clearBinarizedPage,
  exportPage,
  fetchAnnotation,
  fetchHistory,
  fetchTranscription,
  lockPage,
  ocrPage,
  pageImageUrl,
  restoreHistorySnapshot,
  saveAnnotation,
  unlockPage,
} from "@/lib/api";
import { latestHistorySnapshotId } from "@/lib/historyRestore";
import { computePairingProgress, isPairingComplete } from "@/lib/pairingProgress";
import type {
  DrawTool,
  ExportProgressEvent,
  ExportStep,
  HistorySnapshotSummary,
  OcrProgressEvent,
  PageAnnotation,
  Segment,
  TranscriptionResponse,
} from "@/types/api";

interface PageEditorProps {
  stem: string;
  initialDirty: boolean;
}

function exportStepLabel(step: ExportStep): string {
  if (step === "rectify") return "Rectifying";
  return "Saving";
}

function renumberSegments(segments: Segment[]): Segment[] {
  return segments.map((s, i) => ({ ...s, number: i + 1 }));
}

function firstUnpairedLineIndex(
  textLines: TranscriptionResponse["text_lines"],
  segments: Segment[],
  excludeSegmentId?: string,
): number | null {
  const used = new Set(
    segments
      .filter((s) => s.id !== excludeSegmentId && s.paired_text_line_index != null)
      .map((s) => s.paired_text_line_index!),
  );
  const next = textLines.find((l) => !used.has(l.index));
  return next?.index ?? null;
}

export default function PageEditor({ stem, initialDirty }: PageEditorProps) {
  const [annotation, setAnnotation] = useState<PageAnnotation>({
    segments: [],
    export_metadata: null,
    locked: false,
  });
  const [transcription, setTranscription] = useState<TranscriptionResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [transcriptionPromptId, setTranscriptionPromptId] = useState<string | null>(null);
  const [tool, setTool] = useState<DrawTool>("pan");
  const [editMode, setEditMode] = useState(false);
  const [selectedVertexIndex, setSelectedVertexIndex] = useState<number | null>(null);
  const [showSegments, setShowSegments] = useState(true);
  const [showKrakenCeiling, setShowKrakenCeiling] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 1200, height: 1600 });
  const [dirty, setDirty] = useState(initialDirty);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [segmenting, setSegmenting] = useState(false);
  const [binarizing, setBinarizing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgressEvent | null>(null);
  const [ocrRunning, setOcrRunning] = useState(false);
  const [ocrProgress, setOcrProgress] = useState<OcrProgressEvent | null>(null);
  const [toast, setToast] = useState<{ text: string; kind: "success" | "error" } | null>(null);
  const [previewSegmentId, setPreviewSegmentId] = useState<string | null>(null);
  const [pdfPanelMode, setPdfPanelMode] = useState<TranscriptionPdfMode | null>(null);
  const [pdfRefreshKey, setPdfRefreshKey] = useState(0);
  const pdfPanelModeRef = useRef(pdfPanelMode);
  const [showHistory, setShowHistory] = useState(false);
  const [historySnapshots, setHistorySnapshots] = useState<HistorySnapshotSummary[]>([]);
  const [restoring, setRestoring] = useState(false);
  const [showLockPrompt, setShowLockPrompt] = useState(false);
  const lockPromptDismissedRef = useRef(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const canvasRef = useRef<ImageCanvasHandle>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const annotationRef = useRef(annotation);
  const transcriptionRef = useRef(transcription);
  const selectedIdRef = useRef(selectedId);
  const editModeRef = useRef(editMode);
  const selectedVertexIndexRef = useRef(selectedVertexIndex);
  annotationRef.current = annotation;
  transcriptionRef.current = transcription;
  selectedIdRef.current = selectedId;
  editModeRef.current = editMode;
  selectedVertexIndexRef.current = selectedVertexIndex;

  useEffect(() => {
    setSelectedVertexIndex(null);
  }, [selectedId, editMode]);

  useEffect(() => {
    Promise.all([fetchAnnotation(stem), fetchTranscription(stem)]).then(([ann, tx]) => {
      setAnnotation(ann);
      setTranscription(tx);
    });
  }, [stem]);

  useEffect(() => {
    const img = new Image();
    img.onload = () => setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    img.src = pageImageUrl(stem, annotation.binarized_at ?? null);
  }, [stem, annotation.binarized_at]);

  const showToast = useCallback((text: string, kind: "success" | "error" = "success") => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToast({ text, kind });
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  }, []);

  useEffect(() => {
    return () => {
      if (toastTimer.current) clearTimeout(toastTimer.current);
    };
  }, []);

  const locked = annotation.locked;

  const flushSave = useCallback(async (): Promise<PageAnnotation> => {
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    const pending = annotationRef.current;
    if (pending.locked) return pending;
    try {
      const saved = await saveAnnotation(stem, pending);
      setAnnotation(saved);
      setSaveError(null);
      setPdfRefreshKey((k) => k + 1);
      return saved;
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
      throw err;
    }
  }, [stem]);

  const scheduleSave = useCallback(
    (next: PageAnnotation) => {
      if (next.locked) return;
      setAnnotation(next);
      setDirty(true);
      setSaveError(null);
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(async () => {
        try {
          await saveAnnotation(stem, next);
          setPdfRefreshKey((k) => k + 1);
        } catch (err) {
          setSaveError(err instanceof Error ? err.message : "Save failed");
        }
      }, 400);
    },
    [stem],
  );

  const handleAddSegment = useCallback(
    (segment: Segment) => {
      const tx = transcriptionRef.current;
      const ann = annotationRef.current;
      let enriched = segment;

      if (tx && tx.status !== "missing" && tx.text_lines.length > 0) {
        const nextLine = firstUnpairedLineIndex(tx.text_lines, ann.segments);
        if (nextLine != null) {
          enriched = { ...segment, paired_text_line_index: nextLine, text_override: null };
        }
      }

      scheduleSave({ ...ann, segments: [...ann.segments, enriched] });
      setSelectedId(enriched.id);
      setTranscriptionPromptId(enriched.id);
      setTool("select");
      setEditMode(false);
      setShowSegments(true);
    },
    [scheduleSave],
  );

  const handleUpdateSegment = (segment: Segment) => {
    scheduleSave({
      ...annotation,
      segments: annotation.segments.map((s) => (s.id === segment.id ? segment : s)),
    });
  };

  const deleteSelectedVertex = useCallback((): boolean => {
    if (!editModeRef.current) return false;
    const id = selectedIdRef.current;
    const vtx = selectedVertexIndexRef.current;
    if (id == null || vtx == null) return false;
    const ann = annotationRef.current;
    const seg = ann.segments.find((s) => s.id === id);
    if (!seg || seg.points.length <= MIN_SEGMENT_POINTS) return false;
    scheduleSave({
      ...ann,
      segments: ann.segments.map((s) =>
        s.id === id ? { ...s, points: s.points.filter((_, i) => i !== vtx) } : s,
      ),
    });
    setSelectedVertexIndex(null);
    return true;
  }, [scheduleSave]);

  const deleteSelectedSegment = useCallback(() => {
    const id = selectedIdRef.current;
    if (!id) return false;
    const ann = annotationRef.current;
    scheduleSave({
      ...ann,
      segments: renumberSegments(ann.segments.filter((s) => s.id !== id)),
    });
    setSelectedId(null);
    setTranscriptionPromptId(null);
    setEditMode(false);
    return true;
  }, [scheduleSave]);

  const handleSave = useCallback(() => {
    if (locked) return;
    void flushSave().then(() => showToast("Annotation saved."));
  }, [locked, flushSave, showToast]);

  const handleDelete = useCallback(() => {
    if (locked) return;
    if (canvasRef.current?.cancelDraft()) return;
    if (deleteSelectedVertex()) return;
    if (!selectedIdRef.current) return;
    if (!window.confirm("Delete selected segment?")) return;
    deleteSelectedSegment();
  }, [locked, deleteSelectedVertex, deleteSelectedSegment]);

  const handleSelect = (id: string | null) => {
    setSelectedId(id);
    if (id) {
      setEditMode(false);
    } else {
      setTranscriptionPromptId(null);
      setEditMode(false);
    }
  };

  const handlePair = (textLineIndex: number) => {
    if (!selectedId) return;
    const alreadyUsed = annotation.segments.some(
      (s) => s.id !== selectedId && s.paired_text_line_index === textLineIndex,
    );
    if (alreadyUsed) {
      window.alert("That text line is already paired to another segment.");
      return;
    }
    scheduleSave({
      ...annotation,
      segments: annotation.segments.map((s) =>
        s.id === selectedId
          ? { ...s, paired_text_line_index: textLineIndex, text_override: null }
          : s,
      ),
    });
    setTranscriptionPromptId(selectedId);
  };

  const handleTextOverride = (text: string) => {
    if (!selectedId) return;
    scheduleSave({
      ...annotation,
      segments: annotation.segments.map((s) =>
        s.id === selectedId ? { ...s, text_override: text } : s,
      ),
    });
  };

  const finishTranscription = useCallback(() => {
    setSelectedId(null);
    setTranscriptionPromptId(null);
    setTool("pan");
    setEditMode(false);
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const data = await fetchHistory(stem);
      setHistorySnapshots(data.snapshots);
    } catch {
      setHistorySnapshots([]);
    }
  }, [stem]);

  const handleUndo = useCallback(async () => {
    if (canvasRef.current?.cancelDraft()) return;
    if (locked) return;
    try {
      const data = await fetchHistory(stem);
      const snapshotId = latestHistorySnapshotId(data.snapshots);
      if (!snapshotId) {
        showToast("No history to restore.", "error");
        return;
      }
      const restored = await restoreHistorySnapshot(stem, snapshotId);
      setAnnotation(restored);
      setDirty(true);
      setSelectedId(null);
      setTranscriptionPromptId(null);
      setEditMode(false);
      showToast("Restored from history.");
      if (showHistory) await refreshHistory();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Restore failed", "error");
    }
  }, [locked, stem, showHistory, refreshHistory, showToast]);

  useEffect(() => {
    if (showHistory) refreshHistory();
  }, [showHistory, refreshHistory]);

  const handleLock = async () => {
    try {
      await flushSave();
      const result = await lockPage(stem);
      setAnnotation(result);
      setShowLockPrompt(false);
      lockPromptDismissedRef.current = true;
      setPdfRefreshKey((k) => k + 1);
      showToast("Page locked.");
      if (showHistory) refreshHistory();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Lock failed", "error");
    }
  };

  const handleUnlock = async () => {
    if (!window.confirm("Unlock this page for editing?")) return;
    try {
      const result = await unlockPage(stem);
      setAnnotation(result);
      lockPromptDismissedRef.current = false;
      if (pdfPanelMode === "share") setPdfPanelMode("preview");
      setPdfRefreshKey((k) => k + 1);
      showToast("Page unlocked.");
      if (showHistory) refreshHistory();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Unlock failed", "error");
    }
  };

  const handleRestoreSnapshot = async (snapshotId: string) => {
    setRestoring(true);
    try {
      const restored = await restoreHistorySnapshot(stem, snapshotId);
      setAnnotation(restored);
      setDirty(true);
      showToast("Annotation restored from history.");
      await refreshHistory();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Restore failed", "error");
    } finally {
      setRestoring(false);
    }
  };

  const handleAutoSegment = async () => {
    if (locked) return;
    const ann = annotationRef.current;
    if (ann.segments.length > 0) {
      const ok = window.confirm(
        "Run Kraken line segmentation? This will replace all existing segments on this page.",
      );
      if (!ok) return;
    }

    setSegmenting(true);
    try {
      const result = await autoSegmentPage(stem, { replace: true, pair_transcription: true });
      setAnnotation(result);
      setDirty(true);
      setSelectedId(null);
      setTranscriptionPromptId(null);
      setEditMode(false);
      setTool("select");
      setShowSegments(true);
      showToast(`Kraken found ${result.segments.length} line segment(s).`);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Auto-segment failed", "error");
    } finally {
      setSegmenting(false);
    }
  };

  const handleBinarize = async () => {
    if (locked) return;
    setBinarizing(true);
    try {
      const result = await binarizePage(stem);
      setAnnotation(result);
      showToast("Page binarized with Kraken nlbin.");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Binarize failed", "error");
    } finally {
      setBinarizing(false);
    }
  };

  const handleClearBinarize = async () => {
    if (locked) return;
    setBinarizing(true);
    try {
      const result = await clearBinarizedPage(stem);
      setAnnotation(result);
      showToast("Showing original page image.");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Revert failed", "error");
    } finally {
      setBinarizing(false);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    setExportProgress(null);
    try {
      const result = await exportPage(stem, (progress) => setExportProgress(progress));
      setDirty(false);
      const w = result.warnings;
      const unpaired = w.unpaired_segments.join(", ") || "none";
      showToast(`Exported ${result.exported_count} line(s). Unpaired segments: ${unpaired}.`);
      const ann = await fetchAnnotation(stem);
      setAnnotation(ann);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Export failed", "error");
    } finally {
      setExporting(false);
      setExportProgress(null);
    }
  };

  const handleOcrPage = async () => {
    setOcrRunning(true);
    setOcrProgress(null);
    try {
      const result = await ocrPage(stem, (progress) => setOcrProgress(progress));
      const ann = await fetchAnnotation(stem);
      setAnnotation(ann);
      showToast(`OCR completed for ${result.processed_count} segment(s).`);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Page OCR failed", "error");
    } finally {
      setOcrRunning(false);
      setOcrProgress(null);
    }
  };

  const handleOcrSegmentComplete = (segment: Segment) => {
    setAnnotation((ann) => ({
      ...ann,
      segments: ann.segments.map((s) => (s.id === segment.id ? segment : s)),
    }));
  };

  const pickTool = useCallback(
    (next: DrawTool) => {
      if (locked) return;
      setTool(next);
      setEditMode(false);
    },
    [locked],
  );

  const toggleEdit = useCallback(() => {
    if (locked || !selectedIdRef.current) return;
    setTool("select");
    setEditMode((v) => !v);
  }, [locked]);

  const toggleLines = useCallback(() => {
    setShowSegments((v) => !v);
  }, []);

  useEditorShortcuts(
    useMemo(
      () => ({
        onTool: pickTool,
        onToggleEdit: toggleEdit,
        onToggleLines: toggleLines,
        onUndo: () => void handleUndo(),
        onDelete: handleDelete,
        onSave: handleSave,
        onZoomIn: () => canvasRef.current?.zoomIn(),
        onZoomOut: () => canvasRef.current?.zoomOut(),
        onFitPage: () => canvasRef.current?.fitPage(),
      }),
      [pickTool, toggleEdit, toggleLines, handleUndo, handleDelete, handleSave],
    ),
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;

      const inField =
        e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement;

      if (!inField && canvasRef.current?.cancelDraft()) {
        e.preventDefault();
        return;
      }

      if (editModeRef.current && selectedVertexIndexRef.current != null) {
        e.preventDefault();
        setSelectedVertexIndex(null);
        return;
      }

      if (selectedIdRef.current) {
        e.preventDefault();
        setSelectedId(null);
        setTranscriptionPromptId(null);
        setEditMode(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const selectedSegment = annotation.segments.find((s) => s.id === selectedId) ?? null;
  const previewSegment = annotation.segments.find((s) => s.id === previewSegmentId) ?? null;
  const textLines = useMemo(
    () => (transcription?.status === "missing" ? [] : (transcription?.text_lines ?? [])),
    [transcription],
  );
  const pairingProgress = useMemo(
    () => computePairingProgress(annotation.segments, textLines),
    [annotation.segments, textLines],
  );

  const pairingComplete =
    annotation.segments.length > 0 && pairingProgress.unpaired_count === 0;
  const verified =
    annotation.segments.length > 0 && isPairingComplete(pairingProgress);

  useEffect(() => {
    if (!pairingComplete) {
      lockPromptDismissedRef.current = false;
      setShowLockPrompt(false);
      return;
    }
    if (!locked && !lockPromptDismissedRef.current) {
      setShowLockPrompt(true);
    }
  }, [pairingComplete, locked]);

  const pdfPanelOpen = pdfPanelMode !== null;
  const { trailingWidth: pdfPanelWidth, dividerProps } = useHorizontalSplit(splitContainerRef, {
    enabled: pdfPanelOpen,
  });

  useEffect(() => {
    if (pdfPanelModeRef.current === pdfPanelMode) return;
    pdfPanelModeRef.current = pdfPanelMode;
    const id = window.setTimeout(() => canvasRef.current?.fitPage(), 50);
    return () => window.clearTimeout(id);
  }, [pdfPanelMode]);

  useEffect(() => {
    if (!pdfPanelOpen) return;
    const id = window.setTimeout(() => canvasRef.current?.fitPage(), 50);
    return () => window.clearTimeout(id);
  }, [pdfPanelOpen, pdfPanelWidth]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white text-gray-900">
      <EditorHeader
        stem={stem}
        dirty={dirty}
        locked={locked}
        binarized={annotation.binarized_at != null}
        verified={verified}
        pairingProgress={pairingProgress}
        tool={tool}
        editMode={editMode}
        showSegments={showSegments}
        showKrakenCeiling={showKrakenCeiling}
        hasSelection={selectedSegment != null}
        canToggleCeiling={
          selectedSegment != null && (selectedSegment.source ?? "manual") === "kraken"
        }
        binarizing={binarizing}
        segmenting={segmenting}
        ocrRunning={ocrRunning}
        exporting={exporting}
        hasSegments={annotation.segments.length > 0}
        showHistory={showHistory}
        pdfPanelOpen={pdfPanelMode !== null}
        pdfPanelMode={pdfPanelMode}
        onPickTool={pickTool}
        onToggleEdit={toggleEdit}
        onToggleLines={toggleLines}
        onToggleCeiling={() => setShowKrakenCeiling((v) => !v)}
        onDelete={handleDelete}
        onToggleHistory={() => setShowHistory((v) => !v)}
        onBinarize={handleBinarize}
        onClearBinarize={handleClearBinarize}
        onAutoSegment={handleAutoSegment}
        onOcrPage={handleOcrPage}
        onExport={handleExport}
        onLock={handleLock}
        onUnlock={handleUnlock}
        onPdfOpen={(mode) => {
          setPdfRefreshKey((k) => k + 1);
          setPdfPanelMode(mode);
        }}
        onPdfClose={() => setPdfPanelMode(null)}
      />

      {showHistory && (
        <div className="shrink-0 border-b border-gray-100 bg-gray-50 px-3 py-2">
          <div className="mb-1 flex items-center justify-between">
            <h2 className="text-xs font-medium uppercase tracking-wide text-gray-500">History</h2>
            <button
              type="button"
              onClick={() => setShowHistory(false)}
              className="text-xs text-gray-500 hover:text-gray-800"
            >
              Close
            </button>
          </div>
          {locked && (
            <p className="mb-2 text-xs text-amber-800">Unlock the page to restore a snapshot.</p>
          )}
          <HistoryPanel
            snapshots={historySnapshots}
            restoring={restoring}
            locked={locked}
            onRestore={handleRestoreSnapshot}
          />
        </div>
      )}

      {showLockPrompt && !locked && (
        <div className="shrink-0 border-b border-emerald-100 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
          <span>Pairing is complete. Lock this page to freeze annotation?</span>
          <div className="mt-1 flex gap-2">
            <button
              type="button"
              onClick={handleLock}
              className="rounded bg-emerald-800 px-2.5 py-1 text-xs text-white hover:bg-emerald-900"
            >
              Lock page
            </button>
            <button
              type="button"
              onClick={() => {
                lockPromptDismissedRef.current = true;
                setShowLockPrompt(false);
              }}
              className="rounded px-2.5 py-1 text-xs text-emerald-800 hover:bg-emerald-100"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {saveError && <div className="shrink-0 bg-red-50 px-3 py-1.5 text-sm text-red-700">{saveError}</div>}

      {(segmenting || binarizing || exporting || ocrRunning) && (
        <div
          role="alert"
          aria-live="polite"
          className="pointer-events-none fixed left-1/2 top-4 z-50 w-[min(24rem,calc(100vw-2rem))] -translate-x-1/2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 shadow-lg"
        >
          {binarizing ? (
            <>
              <p className="text-sm text-blue-900">Binarizing page with Kraken nlbin…</p>
              <div className="progress-indeterminate mt-2 h-1.5 overflow-hidden rounded-full bg-blue-100" />
            </>
          ) : segmenting ? (
            <>
              <p className="text-sm text-blue-900">Running Kraken line segmentation…</p>
              <div className="progress-indeterminate mt-2 h-1.5 overflow-hidden rounded-full bg-blue-100" />
            </>
          ) : ocrRunning && ocrProgress && ocrProgress.total > 0 ? (
            <>
              <div className="flex items-center justify-between gap-3 text-sm text-blue-900">
                <span>
                  OCR line {ocrProgress.current} of {ocrProgress.total}
                </span>
                <span className="text-xs text-blue-700">segment {ocrProgress.segment_number}</span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-blue-100">
                <div
                  className="h-full rounded-full bg-blue-600 transition-all duration-200"
                  style={{
                    width: `${Math.round((ocrProgress.current / ocrProgress.total) * 100)}%`,
                  }}
                />
              </div>
            </>
          ) : ocrRunning ? (
            <>
              <p className="text-sm text-blue-900">Preparing page OCR…</p>
              <div className="progress-indeterminate mt-2 h-1.5 overflow-hidden rounded-full bg-blue-100" />
            </>
          ) : exportProgress && exportProgress.total > 0 ? (
            <>
              <div className="flex items-center justify-between gap-3 text-sm text-blue-900">
                <span>
                  {exportStepLabel(exportProgress.step)} line {exportProgress.current} of{" "}
                  {exportProgress.total}
                </span>
                <span className="text-xs text-blue-700">segment {exportProgress.segment_number}</span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-blue-100">
                <div
                  className="h-full rounded-full bg-blue-600 transition-all duration-200"
                  style={{
                    width: `${Math.round((exportProgress.current / exportProgress.total) * 100)}%`,
                  }}
                />
              </div>
            </>
          ) : (
            <>
              <p className="text-sm text-blue-900">Preparing export…</p>
              <div className="progress-indeterminate mt-2 h-1.5 overflow-hidden rounded-full bg-blue-100" />
            </>
          )}
        </div>
      )}

      {toast && !segmenting && !binarizing && !exporting && !ocrRunning && (
        <div
          role="alert"
          className={`pointer-events-none fixed left-1/2 top-4 z-50 -translate-x-1/2 rounded-lg border px-4 py-2.5 text-sm shadow-lg ${
            toast.kind === "error"
              ? "border-red-200 bg-red-50 text-red-800"
              : "border-gray-200 bg-white text-gray-800"
          }`}
        >
          {toast.text}
        </div>
      )}

      <div ref={splitContainerRef} className="flex min-h-0 flex-1">
        <div
          className="flex min-h-0 min-w-0 flex-col"
          style={pdfPanelOpen && pdfPanelWidth != null ? { flex: "1 1 0" } : { flex: "1 1 auto" }}
        >
          <div className="relative min-h-0 flex-1">
            <ImageCanvas
              ref={canvasRef}
              imageUrl={pageImageUrl(stem, annotation.binarized_at ?? null)}
              imageWidth={imageSize.width}
              imageHeight={imageSize.height}
              segments={annotation.segments}
              selectedId={selectedId}
              tool={locked ? "pan" : tool}
              editMode={locked ? false : editMode}
              readOnly={locked}
              showSegments={showSegments}
              showKrakenCeiling={showKrakenCeiling}
              selectedVertexIndex={selectedVertexIndex}
              onSelectVertex={setSelectedVertexIndex}
              onSelect={handleSelect}
              onAddSegment={handleAddSegment}
              onUpdateSegment={handleUpdateSegment}
            />
          </div>

          {selectedSegment && (
            <SegmentPairingBar
              stem={stem}
              segment={selectedSegment}
              textLines={textLines}
              segments={annotation.segments}
              locked={locked}
              autoFocus={!locked && selectedSegment.id === transcriptionPromptId}
              onPair={handlePair}
              onTextOverride={handleTextOverride}
              onOcrComplete={handleOcrSegmentComplete}
              onOcrError={(message) => showToast(message, "error")}
              onSave={() => void flushSave()}
              onClose={() => handleSelect(null)}
              onDone={finishTranscription}
              onPreviewExport={() => setPreviewSegmentId(selectedSegment.id)}
            />
          )}
        </div>

        {pdfPanelOpen && pdfPanelWidth != null && (
          <>
            <SplitPaneDivider {...dividerProps} />
            <div className="flex min-h-0 shrink-0 flex-col" style={{ width: pdfPanelWidth }}>
              <TranscriptionPdfPanel
                stem={stem}
                mode={pdfPanelMode!}
                locked={locked}
                refreshKey={pdfRefreshKey}
                onClose={() => setPdfPanelMode(null)}
                onSwitchMode={(mode) => {
                  if (mode === "share" && !locked) return;
                  setPdfRefreshKey((k) => k + 1);
                  setPdfPanelMode(mode);
                }}
              />
            </div>
          </>
        )}
      </div>

      {previewSegment && (
        <ExportPreviewModal
          stem={stem}
          segmentId={previewSegment.id}
          segmentNumber={previewSegment.number}
          onClose={() => setPreviewSegmentId(null)}
        />
      )}
    </div>
  );
}
