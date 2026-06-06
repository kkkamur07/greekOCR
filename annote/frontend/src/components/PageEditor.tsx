"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ImageCanvas, { type ImageCanvasHandle } from "@/components/ImageCanvas/ImageCanvas";
import SegmentPairingBar from "@/components/SegmentPairingBar";
import { EDITOR_SHORTCUTS, useEditorShortcuts } from "@/hooks/useEditorShortcuts";
import {
  exportPage,
  fetchAnnotation,
  fetchTranscription,
  pageImageUrl,
  saveAnnotation,
} from "@/lib/api";
import { displayPageName, formatPageTitle } from "@/lib/pageName";
import type {
  DrawTool,
  ExportProgressEvent,
  ExportStep,
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
  if (step === "binarize") return "Binarizing";
  if (step === "rectify") return "Rectifying";
  return "Saving";
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
  const [annotation, setAnnotation] = useState<PageAnnotation>({ segments: [], export_metadata: null });
  const [transcription, setTranscription] = useState<TranscriptionResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [transcriptionPromptId, setTranscriptionPromptId] = useState<string | null>(null);
  const [tool, setTool] = useState<DrawTool>("pan");
  const [editMode, setEditMode] = useState(false);
  const [showSegments, setShowSegments] = useState(true);
  const [imageSize, setImageSize] = useState({ width: 1200, height: 1600 });
  const [dirty, setDirty] = useState(initialDirty);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgressEvent | null>(null);
  const [binarizeOnExport, setBinarizeOnExport] = useState(false);
  const [toast, setToast] = useState<{ text: string; kind: "success" | "error" } | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const canvasRef = useRef<ImageCanvasHandle>(null);
  const annotationRef = useRef(annotation);
  const transcriptionRef = useRef(transcription);
  const selectedIdRef = useRef(selectedId);
  annotationRef.current = annotation;
  transcriptionRef.current = transcription;
  selectedIdRef.current = selectedId;

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

  const scheduleSave = useCallback(
    (next: PageAnnotation) => {
      setAnnotation(next);
      setDirty(true);
      setSaveError(null);
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(async () => {
        try {
          await saveAnnotation(stem, next);
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

  const deleteSelectedSegment = useCallback(() => {
    const id = selectedIdRef.current;
    if (!id) return false;
    const ann = annotationRef.current;
    scheduleSave({
      ...ann,
      segments: ann.segments.filter((s) => s.id !== id),
    });
    setSelectedId(null);
    setTranscriptionPromptId(null);
    return true;
  }, [scheduleSave]);

  const handleDeleteKey = useCallback(() => {
    if (canvasRef.current?.cancelDraft()) return;
    deleteSelectedSegment();
  }, [deleteSelectedSegment]);

  const handleDeleteClick = () => {
    if (canvasRef.current?.cancelDraft()) return;
    if (!selectedIdRef.current) return;
    if (!window.confirm("Delete selected segment?")) return;
    deleteSelectedSegment();
  };

  const handleSelect = (id: string | null) => {
    setSelectedId(id);
    if (id) {
      setEditMode(false);
    } else {
      setTranscriptionPromptId(null);
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

  const handleExport = async () => {
    setExporting(true);
    setExportProgress(null);
    try {
      const result = await exportPage(
        stem,
        { binarize: binarizeOnExport },
        (progress) => setExportProgress(progress),
      );
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

  const pickTool = useCallback((next: DrawTool) => {
    setTool(next);
    setEditMode(false);
  }, []);

  const toggleEdit = useCallback(() => {
    setTool("select");
    setEditMode((v) => !v);
  }, []);

  const toggleLines = useCallback(() => {
    setShowSegments((v) => !v);
  }, []);

  useEditorShortcuts(
    useMemo(
      () => ({
        onTool: pickTool,
        onToggleEdit: toggleEdit,
        onToggleLines: toggleLines,
        onDelete: handleDeleteKey,
        onZoomIn: () => canvasRef.current?.zoomIn(),
        onZoomOut: () => canvasRef.current?.zoomOut(),
        onFitPage: () => canvasRef.current?.fitPage(),
      }),
      [pickTool, toggleEdit, toggleLines, handleDeleteKey],
    ),
  );

  const selectedSegment = annotation.segments.find((s) => s.id === selectedId) ?? null;
  const textLines = transcription?.status === "missing" ? [] : (transcription?.text_lines ?? []);

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
            className={toolBtn(tool === "select" && !editMode)}
            title={`Select (${EDITOR_SHORTCUTS.select})`}
          >
            Select <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.select}</kbd>
          </button>
          <button
            type="button"
            onClick={() => pickTool("rectangle")}
            className={toolBtn(tool === "rectangle")}
            title={`Rectangle (${EDITOR_SHORTCUTS.rectangle})`}
          >
            Rect <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.rectangle}</kbd>
          </button>
          <button
            type="button"
            onClick={() => pickTool("polygon")}
            className={toolBtn(tool === "polygon")}
            title={`Polygon (${EDITOR_SHORTCUTS.polygon})`}
          >
            Poly <kbd className="ml-1 text-[10px] opacity-60">{EDITOR_SHORTCUTS.polygon}</kbd>
          </button>
          <button
            type="button"
            onClick={toggleEdit}
            className={toolBtn(editMode)}
            title={`Edit vertices (${EDITOR_SHORTCUTS.editVertices})`}
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
            onClick={handleDeleteClick}
            className="rounded px-2.5 py-1 text-sm text-red-700 hover:bg-red-50"
            title="Delete (Del)"
          >
            Del
          </button>
          <label
            className="ml-2 flex cursor-pointer items-center gap-1.5 text-xs text-gray-600"
            title="Kraken nlbin — requires pip install 'annote[kraken]'"
          >
            <input
              type="checkbox"
              checked={binarizeOnExport}
              onChange={(e) => setBinarizeOnExport(e.target.checked)}
              className="rounded border-gray-300"
            />
            Binarize
          </label>
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

      {exporting && (
        <div className="shrink-0 border-b border-blue-100 bg-blue-50 px-3 py-2">
          {exportProgress && exportProgress.total > 0 ? (
            <>
              <div className="flex items-center justify-between gap-3 text-sm text-blue-900">
                <span>
                  {exportStepLabel(exportProgress.step)} line {exportProgress.current} of{" "}
                  {exportProgress.total}
                  {exportProgress.step === "binarize" ? " (Kraken)" : ""}
                </span>
                <span className="text-xs text-blue-700">segment {exportProgress.segment_number}</span>
              </div>
              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-blue-100">
                <div
                  className="h-full rounded-full bg-blue-600 transition-all duration-200"
                  style={{
                    width: `${Math.round((exportProgress.current / exportProgress.total) * 100)}%`,
                  }}
                />
              </div>
            </>
          ) : (
            <p className="text-sm text-blue-900">Preparing export…</p>
          )}
        </div>
      )}

      <div className="shrink-0 border-b border-gray-100 bg-gray-50 px-3 py-1 text-[11px] text-gray-500">
        <span className="font-medium text-gray-600">Shortcuts:</span>{" "}
        Drag to pan (polygon too) · scroll to zoom · Enter finish polygon · Esc cancel draw · Del delete ·{" "}
        {EDITOR_SHORTCUTS.fitPage} fit page
      </div>

      {saveError && <div className="shrink-0 bg-red-50 px-3 py-1.5 text-sm text-red-700">{saveError}</div>}

      {toast && (
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

      <div className="flex min-h-0 flex-1 flex-col">
        <div className="relative min-h-0 flex-1">
          <ImageCanvas
            ref={canvasRef}
            imageUrl={pageImageUrl(stem)}
            imageWidth={imageSize.width}
            imageHeight={imageSize.height}
            segments={annotation.segments}
            selectedId={selectedId}
            tool={tool}
            editMode={editMode}
            showSegments={showSegments}
            onSelect={handleSelect}
            onAddSegment={handleAddSegment}
            onUpdateSegment={handleUpdateSegment}
          />
        </div>

        {selectedSegment && (
          <SegmentPairingBar
            segment={selectedSegment}
            textLines={textLines}
            segments={annotation.segments}
            autoFocus={selectedSegment.id === transcriptionPromptId}
            onPair={handlePair}
            onTextOverride={handleTextOverride}
            onClose={() => handleSelect(null)}
            onDone={finishTranscription}
          />
        )}
      </div>
    </div>
  );
}
