"use client";

import { useRef, useState } from "react";
import type { TranscriptionPdfMode } from "@/components/TranscriptionPdfPanel";
import {
  BinarizeIcon,
  OcrIcon,
  PdfIcon,
  SegmentIcon,
  ShareIcon,
  WorkflowIcon,
} from "./icons";
import { MenuRow, MenuSection } from "./MenuPrimitives";
import { useDismissibleMenu } from "./useDismissibleMenu";

interface EditorWorkflowMenuProps {
  locked: boolean;
  binarized: boolean;
  binarizing: boolean;
  segmenting: boolean;
  ocrRunning: boolean;
  exporting: boolean;
  hasSegments: boolean;
  pdfPanelOpen: boolean;
  pdfPanelMode: TranscriptionPdfMode | null;
  onBinarize: () => void;
  onClearBinarize: () => void;
  onAutoSegment: () => void;
  onOcrPage: () => void;
  onPdfOpen: (mode: TranscriptionPdfMode) => void;
  onPdfClose: () => void;
}

export default function EditorWorkflowMenu({
  locked,
  binarized,
  binarizing,
  segmenting,
  ocrRunning,
  exporting,
  hasSegments,
  pdfPanelOpen,
  pdfPanelMode,
  onBinarize,
  onClearBinarize,
  onAutoSegment,
  onOcrPage,
  onPdfOpen,
  onPdfClose,
}: EditorWorkflowMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useDismissibleMenu(open, () => setOpen(false), rootRef);

  const busy = binarizing || segmenting || ocrRunning || exporting;
  const workflowActive = pdfPanelOpen || binarized || open;

  const selectPdfMode = (mode: TranscriptionPdfMode) => {
    setOpen(false);
    if (mode === "share" && !locked) return;
    if (pdfPanelOpen && pdfPanelMode === mode) {
      onPdfClose();
      return;
    }
    onPdfOpen(mode);
  };

  const runAndClose = (fn: () => void) => {
    setOpen(false);
    fn();
  };

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium ring-1 ${
          workflowActive
            ? "bg-indigo-50 text-indigo-900 ring-indigo-200"
            : "bg-white text-gray-800 ring-gray-200 hover:bg-gray-50"
        }`}
      >
        <WorkflowIcon className="h-3.5 w-3.5 opacity-80" />
        Workflow
        <span className="text-[10px] opacity-50" aria-hidden>
          ▾
        </span>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-1 w-64 rounded-lg border border-gray-200 bg-white p-1.5 shadow-xl"
        >
          <MenuSection label="Pipeline">
            <MenuRow
              icon={<BinarizeIcon />}
              label={binarizing ? "Binarizing…" : binarized ? "Show original" : "Binarize"}
              hint="Kraken nlbin"
              active={binarized}
              disabled={busy || locked}
              onClick={() => runAndClose(binarized ? onClearBinarize : onBinarize)}
            />
            <MenuRow
              icon={<SegmentIcon />}
              label={segmenting ? "Segmenting…" : "Auto segment"}
              hint="Kraken BLLA lines"
              disabled={busy || locked}
              onClick={() => runAndClose(onAutoSegment)}
            />
            <MenuRow
              icon={<OcrIcon />}
              label={ocrRunning ? "OCR running…" : "OCR page"}
              hint="Calamari on all segments"
              disabled={busy || !hasSegments}
              onClick={() => runAndClose(onOcrPage)}
            />
          </MenuSection>

          <hr className="my-1 border-gray-100" />

          <MenuSection label="Output">
            <MenuRow
              icon={<PdfIcon />}
              label="Transcription PDF"
              hint="Live preview beside manuscript"
              active={pdfPanelOpen && pdfPanelMode === "preview"}
              onClick={() => selectPdfMode("preview")}
            />
            <MenuRow
              icon={<ShareIcon />}
              label="Share PDF"
              hint={locked ? "Frozen PDF from lock time" : "Lock page first"}
              disabled={!locked}
              onClick={() => selectPdfMode("share")}
            />
          </MenuSection>
        </div>
      )}
    </div>
  );
}
