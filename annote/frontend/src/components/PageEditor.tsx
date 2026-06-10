"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ImageCanvas, { type ImageCanvasHandle } from "@/components/ImageCanvas/ImageCanvas";
import { MIN_SEGMENT_POINTS } from "@/components/ImageCanvas/SegmentOverlay";
import ExportPreviewModal from "@/components/ExportPreviewModal";
import HistoryPanel from "@/components/HistoryPanel";
import PairingProgressBar from "@/components/PairingProgressBar";
import SegmentPairingBar from "@/components/SegmentPairingBar";
import TranscriptionPdfMenu from "@/components/TranscriptionPdfMenu";
import TranscriptionPdfPanel, { type TranscriptionPdfMode } from "@/components/TranscriptionPdfPanel";
import { EDITOR_SHORTCUTS, useEditorShortcuts } from "@/hooks/useEditorShortcuts";
import {
  autoSegmentPage,
  exportPage,
  fetchAnnotation,
  fetchHistory,
  fetchTranscription,
  lockPage,
  pageImageUrl,
  restoreHistorySnapshot,
  saveAnnotation,
  unlockPage,
} from "@/lib/api";
import { latestHistorySnapshotId } from "@/lib/historyRestore";
import { computePairingProgress } from "@/lib/pairingProgress";
import { displayPageName, formatPageTitle } from "@/lib/pageName";
import type {
  DrawTool,
  ExportProgressEvent,
  ExportStep,
  HistorySnapshotSummary,
  PageAnnotation,
  Segment,
  TranscriptionResponse,
} from "@/types/api";

interface PageEditorProps {
  stem: string;
  initialDirty: boolean;
}

function toolBtn(active: boolean) {
  return active
    ? "rounded bg-gray-900 px-2.5 py-1 text-sm text-white"
    : "rounded px-2.5 py-1 text-sm text-gray-700 hover:bg-gray-100";
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
  const [imageSize, setImageSize] = useState({ width: 1200, height: 1600 });
  const [dirty, setDirty] = useState(initialDirty);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [segmenting, setSegmenting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgressEvent | null>(null);
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
    img.src = pageImageUrl(stem);
  }, [stem]);

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

  useEffect(() => {
    if (pdfPanelModeRef.current === pdfPanelMode) return;
    pdfPanelModeRef.current = pdfPanelMode;
    const id = window.setTimeout(() => canvasRef.current?.fitPage(), 50);
    return () => window.clearTimeout(id);
  }, [pdfPanelMode]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white text-gray-900">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-gray-200 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <Link href="/" className="shrink-0 text-sm text-gray-500 hover:text-gray-800">
            ← Back
          </Link>
          <h1 className="truncate text-base font-medium" title={formatPageTitle(stem)}>
            {displayPageName(stem)}
          </h1>
          {dirty && <span className="shrink-0 text-xs text-amber-700">unsaved export</span>}
          {locked && (
            <span className="shrink-0 rounded-full bg-slate-200 px-2 py-0.5 text-xs font-medium text-slate-700">
              locked
            </span>
          )}
        </div>

        <div className="hidden min-w-0 flex-1 justify-center px-2 md:flex">
          {annotation.segments.length > 0 && <PairingProgressBar progress={pairingProgress} />}
        </div>

        <div className="flex shrink-0 items-center gap-0.5">
          <button
            type="button"
            onClick={() => pickTool("pan")}
            className={toolBtn(tool === "pan")}
            title={`Pan (${EDITOR_SHORTCUTS.pan})`}
          >
            Pan <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.pan}</kbd>
          </button>
          <button
            type="button"
            onClick={() => pickTool("select")}
            disabled={locked}
            className={`${toolBtn(tool === "select" && !editMode)} disabled:cursor-not-allowed disabled:opacity-40`}
            title={`Select (${EDITOR_SHORTCUTS.select})`}
          >
            Select <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.select}</kbd>
          </button>
          <button
            type="button"
            onClick={() => pickTool("rectangle")}
            disabled={locked}
            className={`${toolBtn(tool === "rectangle")} disabled:cursor-not-allowed disabled:opacity-40`}
            title={`Rectangle (${EDITOR_SHORTCUTS.rectangle})`}
          >
            Rect <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.rectangle}</kbd>
          </button>
          <button
            type="button"
            onClick={() => pickTool("polygon")}
            disabled={locked}
            className={`${toolBtn(tool === "polygon")} disabled:cursor-not-allowed disabled:opacity-40`}
            title={`Polygon (${EDITOR_SHORTCUTS.polygon}) · ${EDITOR_SHORTCUTS.undo} undo point while drawing`}
          >
            Poly <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.polygon}</kbd>
          </button>
          <button
            type="button"
            onClick={toggleEdit}
            disabled={!selectedSegment || locked}
            className={`${toolBtn(editMode)} disabled:cursor-not-allowed disabled:opacity-40`}
            title={
              selectedSegment
                ? `Edit vertices (${EDITOR_SHORTCUTS.editVertices}) — drag handles, click edge to add point, Del to remove selected point`
                : "Select a segment first"
            }
          >
            Edit <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.editVertices}</kbd>
          </button>
          <button
            type="button"
            onClick={toggleLines}
            className={toolBtn(showSegments)}
            title={`Show lines (${EDITOR_SHORTCUTS.toggleLines})`}
          >
            Lines <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.toggleLines}</kbd>
          </button>
          <button
            type="button"
            onClick={handleAutoSegment}
            disabled={segmenting || exporting || locked}
            className="rounded px-2.5 py-1 text-sm text-indigo-800 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-40"
            title="Auto line segmentation (Kraken BLLA) — requires pip install 'annote[kraken]'"
          >
            Auto segment
          </button>
          <button
            type="button"
            onClick={handleDelete}
            disabled={locked}
            className="rounded px-2.5 py-1 text-sm text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-40"
            title="Delete (Del)"
          >
            Del
          </button>
          <button
            type="button"
            onClick={() => setShowHistory((v) => !v)}
            className="rounded px-2.5 py-1 text-sm text-gray-700 hover:bg-gray-100"
            title="Annotation history"
          >
            History
          </button>
          {locked ? (
            <button
              type="button"
              onClick={handleUnlock}
              className="rounded px-2.5 py-1 text-sm text-amber-800 hover:bg-amber-50"
              title="Unlock page for editing"
            >
              Unlock
            </button>
          ) : (
            <button
              type="button"
              onClick={handleLock}
              className="rounded px-2.5 py-1 text-sm text-slate-700 hover:bg-slate-100"
              title="Lock page to freeze annotation"
            >
              Lock
            </button>
          )}
          <TranscriptionPdfMenu
            locked={locked}
            panelOpen={pdfPanelMode !== null}
            panelMode={pdfPanelMode}
            onOpen={(mode) => {
              setPdfRefreshKey((k) => k + 1);
              setPdfPanelMode(mode);
            }}
            onClose={() => setPdfPanelMode(null)}
          />
          <button
            type="button"
            onClick={handleExport}
            disabled={exporting}
            className="ml-1 rounded bg-gray-900 px-3 py-1 text-sm text-white disabled:opacity-50"
          >
            {exporting ? "Exporting…" : "Export"}
          </button>
        </div>
      </header>

      {annotation.segments.length > 0 && (
        <div className="shrink-0 border-b border-gray-100 bg-gray-50 px-3 py-1.5 md:hidden">
          <PairingProgressBar progress={pairingProgress} />
        </div>
      )}

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

      {(segmenting || exporting) && (
        <div
          role="alert"
          aria-live="polite"
          className="pointer-events-none fixed left-1/2 top-4 z-50 w-[min(24rem,calc(100vw-2rem))] -translate-x-1/2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 shadow-lg"
        >
          {segmenting ? (
            <>
              <p className="text-sm text-blue-900">Running Kraken line segmentation…</p>
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

      {toast && !segmenting && !exporting && (
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

      <div className="flex min-h-0 flex-1">
        <div className={`flex min-h-0 min-w-0 flex-col ${pdfPanelMode ? "w-1/2" : "flex-1"}`}>
          <div className="relative min-h-0 flex-1">
            <ImageCanvas
              ref={canvasRef}
              imageUrl={pageImageUrl(stem)}
              imageWidth={imageSize.width}
              imageHeight={imageSize.height}
              segments={annotation.segments}
              selectedId={selectedId}
              tool={locked ? "pan" : tool}
              editMode={locked ? false : editMode}
              readOnly={locked}
              showSegments={showSegments}
              selectedVertexIndex={selectedVertexIndex}
              onSelectVertex={setSelectedVertexIndex}
              onSelect={handleSelect}
              onAddSegment={handleAddSegment}
              onUpdateSegment={handleUpdateSegment}
            />
          </div>

          {selectedSegment && !locked && (
            <SegmentPairingBar
              segment={selectedSegment}
              textLines={textLines}
              segments={annotation.segments}
              autoFocus={selectedSegment.id === transcriptionPromptId}
              onPair={handlePair}
              onTextOverride={handleTextOverride}
              onSave={() => void flushSave()}
              onClose={() => handleSelect(null)}
              onDone={finishTranscription}
              onPreviewExport={() => setPreviewSegmentId(selectedSegment.id)}
            />
          )}
        </div>

        {pdfPanelMode && (
          <TranscriptionPdfPanel
            stem={stem}
            mode={pdfPanelMode}
            locked={locked}
            refreshKey={pdfRefreshKey}
            onClose={() => setPdfPanelMode(null)}
            onSwitchMode={(mode) => {
              if (mode === "share" && !locked) return;
              setPdfRefreshKey((k) => k + 1);
              setPdfPanelMode(mode);
            }}
          />
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
