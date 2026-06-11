"use client";

import { useEffect, useRef, useState } from "react";
import type { Segment, TextLine } from "@/types/api";

interface SegmentPairingBarProps {
  segment: Segment;
  textLines: TextLine[];
  segments: Segment[];
  stem: string;
  locked?: boolean;
  autoFocus?: boolean;
  onPair: (textLineIndex: number) => void;
  onTextOverride: (text: string) => void;
  onOcrComplete?: (segment: Segment) => void;
  onSave?: () => void;
  onClose: () => void;
  onDone: () => void;
  onPreviewExport?: () => void;
  onOcrError?: (message: string) => void;
}

export default function SegmentPairingBar({
  segment,
  textLines,
  segments,
  stem,
  locked = false,
  autoFocus = false,
  onPair,
  onTextOverride,
  onOcrComplete,
  onSave,
  onClose,
  onDone,
  onPreviewExport,
  onOcrError,
}: SegmentPairingBarProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [ocrLoading, setOcrLoading] = useState(false);
  const pairedLineIndices = new Set(
    segments
      .filter((s) => s.id !== segment.id && s.paired_text_line_index != null)
      .map((s) => s.paired_text_line_index!),
  );

  const pairedLine = textLines.find((l) => l.index === segment.paired_text_line_index);
  const value = segment.text_override ?? pairedLine?.text ?? "";
  const nextUnpaired = textLines.find((l) => !pairedLineIndices.has(l.index));
  const modelSuggestion = segment.model_transcription;

  const handleOcr = async () => {
    setOcrLoading(true);
    try {
      const { runSegmentOcr } = await import("@/lib/api");
      const updated = await runSegmentOcr(stem, segment.id);
      const refreshed = updated.segments.find((s) => s.id === segment.id);
      if (refreshed) onOcrComplete?.(refreshed);
    } catch (err) {
      onOcrError?.(err instanceof Error ? err.message : "OCR failed");
    } finally {
      setOcrLoading(false);
    }
  };

  useEffect(() => {
    if (!autoFocus) return;
    const el = textareaRef.current;
    if (!el) return;
    el.focus();
    if (!value) el.select();
  }, [segment.id, autoFocus, value]);

  return (
    <div className="shrink-0 border-t border-gray-200 bg-white px-4 py-3 shadow-[0_-4px_20px_rgba(0,0,0,0.06)]">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium text-gray-900">
            Segment {segment.number}
            {value ? "" : " — add transcription"}
          </p>
          <div className="flex items-center gap-2">
            {onPreviewExport && (
              <button
                type="button"
                onClick={onPreviewExport}
                className="rounded border border-gray-300 px-3 py-1 text-xs text-gray-700 hover:bg-gray-50"
              >
                Preview export
              </button>
            )}
            <button
              type="button"
              onClick={onDone}
              className="rounded bg-gray-900 px-3 py-1 text-xs text-white hover:bg-gray-800"
            >
              Done · draw next
            </button>
            <button type="button" onClick={onClose} className="text-xs text-gray-500 hover:text-gray-800">
              Close
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => void handleOcr()}
            disabled={ocrLoading}
            className="rounded border border-indigo-300 px-3 py-1 text-xs text-indigo-800 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {ocrLoading ? "OCR…" : "OCR"}
          </button>
          {modelSuggestion != null && modelSuggestion !== "" && (
            <button
              type="button"
              onClick={() => onTextOverride(modelSuggestion)}
              disabled={locked}
              className="rounded border border-gray-300 px-3 py-1 text-xs text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Use suggestion
            </button>
          )}
        </div>

        {modelSuggestion != null && modelSuggestion !== "" && (
          <div className="rounded-lg border border-indigo-100 bg-indigo-50/50 px-3 py-2 text-sm text-indigo-950">
            <p className="text-xs font-medium text-indigo-700">Model suggestion</p>
            <p className="mt-0.5 whitespace-pre-wrap leading-relaxed">{modelSuggestion}</p>
          </div>
        )}

        <textarea
          ref={textareaRef}
          className="w-full resize-y rounded-lg border border-gray-300 p-3 text-sm leading-relaxed focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100 disabled:bg-gray-50 disabled:text-gray-600"
          rows={3}
          value={value}
          disabled={locked}
          onChange={(e) => onTextOverride(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSave?.();
              onDone();
            }
          }}
          placeholder="Type the line transcription here…"
        />

        {textLines.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            <p className="text-xs text-gray-500">Or pick from page transcription:</p>
            <div className="flex flex-wrap gap-1.5">
              {textLines.map((line) => {
                const used = pairedLineIndices.has(line.index);
                const active = segment.paired_text_line_index === line.index;
                const suggested = !active && !used && nextUnpaired?.index === line.index;
                const pairedLine = active || used;
                return (
                  <button
                    key={line.index}
                    type="button"
                    disabled={used && !active}
                    onClick={() => onPair(line.index)}
                    className={`max-w-full truncate rounded-lg border px-2.5 py-1 text-left text-xs ${
                      active
                        ? "border-green-600 bg-green-50 text-green-900"
                        : used
                          ? "border-green-300 bg-green-50/60 text-green-800"
                          : suggested
                            ? "border-amber-400 bg-amber-50 text-amber-900"
                            : "border-amber-300 bg-amber-50/40 text-amber-900 hover:border-amber-500 hover:bg-amber-50"
                    }`}
                    title={line.text}
                  >
                    <span className="mr-1 font-mono text-gray-500">{line.index}.</span>
                    {line.text}
                    {pairedLine && !active && <span className="ml-1 text-green-700">(paired)</span>}
                    {suggested && <span className="ml-1 text-amber-700">(next)</span>}
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <p className="text-xs text-gray-500">
            No page transcription file — type text above. It saves with this segment on export.
          </p>
        )}

        <p className="text-xs text-gray-400">
          Enter to save and draw the next region · Shift+Enter for a new line · Escape to close
        </p>
      </div>
    </div>
  );
}
