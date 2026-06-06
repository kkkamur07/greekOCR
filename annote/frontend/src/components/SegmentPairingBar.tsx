"use client";

import { useEffect, useRef } from "react";
import type { Segment, TextLine } from "@/types/api";

interface SegmentPairingBarProps {
  segment: Segment;
  textLines: TextLine[];
  segments: Segment[];
  autoFocus?: boolean;
  onPair: (textLineIndex: number) => void;
  onTextOverride: (text: string) => void;
  onClose: () => void;
  onDone: () => void;
}

export default function SegmentPairingBar({
  segment,
  textLines,
  segments,
  autoFocus = false,
  onPair,
  onTextOverride,
  onClose,
  onDone,
}: SegmentPairingBarProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pairedLineIndices = new Set(
    segments
      .filter((s) => s.id !== segment.id && s.paired_text_line_index != null)
      .map((s) => s.paired_text_line_index!),
  );

  const pairedLine = textLines.find((l) => l.index === segment.paired_text_line_index);
  const value = segment.text_override ?? pairedLine?.text ?? "";
  const nextUnpaired = textLines.find((l) => !pairedLineIndices.has(l.index));

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

        <textarea
          ref={textareaRef}
          className="w-full resize-y rounded-lg border border-gray-300 p-3 text-sm leading-relaxed focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
          rows={3}
          value={value}
          onChange={(e) => onTextOverride(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
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
                return (
                  <button
                    key={line.index}
                    type="button"
                    disabled={used && !active}
                    onClick={() => onPair(line.index)}
                    className={`max-w-full truncate rounded-lg border px-2.5 py-1 text-left text-xs ${
                      active
                        ? "border-blue-500 bg-blue-50 text-blue-900"
                        : suggested
                          ? "border-amber-400 bg-amber-50 text-amber-900"
                          : used
                            ? "border-gray-200 text-gray-400"
                            : "border-gray-300 hover:border-blue-400 hover:bg-blue-50"
                    }`}
                    title={line.text}
                  >
                    <span className="mr-1 font-mono text-gray-500">{line.index}.</span>
                    {line.text}
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

        <p className="text-xs text-gray-400">Enter to finish and draw the next region · Shift+Enter for a new line</p>
      </div>
    </div>
  );
}
