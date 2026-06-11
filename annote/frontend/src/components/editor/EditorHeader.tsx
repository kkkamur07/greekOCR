"use client";

import Link from "next/link";
import type { TranscriptionPdfMode } from "@/components/TranscriptionPdfPanel";
import { formatPairingProgress } from "@/lib/pairingProgress";
import { displayPageName, formatPageTitle } from "@/lib/pageName";
import type { DrawTool, PairingProgress } from "@/types/api";
import EditorMoreMenu from "./EditorMoreMenu";
import EditorToolStrip from "./EditorToolStrip";
import EditorWorkflowMenu from "./EditorWorkflowMenu";

interface EditorHeaderProps {
  stem: string;
  dirty: boolean;
  locked: boolean;
  binarized: boolean;
  verified: boolean;
  pairingProgress: PairingProgress;
  tool: DrawTool;
  editMode: boolean;
  showSegments: boolean;
  showKrakenCeiling: boolean;
  hasSelection: boolean;
  canToggleCeiling: boolean;
  binarizing: boolean;
  segmenting: boolean;
  ocrRunning: boolean;
  exporting: boolean;
  hasSegments: boolean;
  showHistory: boolean;
  pdfPanelOpen: boolean;
  pdfPanelMode: TranscriptionPdfMode | null;
  onPickTool: (tool: DrawTool) => void;
  onToggleEdit: () => void;
  onToggleLines: () => void;
  onToggleCeiling: () => void;
  onDelete: () => void;
  onToggleHistory: () => void;
  onBinarize: () => void;
  onClearBinarize: () => void;
  onAutoSegment: () => void;
  onOcrPage: () => void;
  onExport: () => void;
  onLock: () => void;
  onUnlock: () => void;
  onPdfOpen: (mode: TranscriptionPdfMode) => void;
  onPdfClose: () => void;
}

export default function EditorHeader({
  stem,
  dirty,
  locked,
  binarized,
  verified,
  pairingProgress,
  tool,
  editMode,
  showSegments,
  showKrakenCeiling,
  hasSelection,
  canToggleCeiling,
  binarizing,
  segmenting,
  ocrRunning,
  exporting,
  hasSegments,
  showHistory,
  pdfPanelOpen,
  pdfPanelMode,
  onPickTool,
  onToggleEdit,
  onToggleLines,
  onToggleCeiling,
  onDelete,
  onToggleHistory,
  onBinarize,
  onClearBinarize,
  onAutoSegment,
  onOcrPage,
  onExport,
  onLock,
  onUnlock,
  onPdfOpen,
  onPdfClose,
}: EditorHeaderProps) {
  return (
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
        {binarized && (
          <span className="shrink-0 rounded-full bg-zinc-200 px-2 py-0.5 text-xs font-medium text-zinc-700">
            binarized
          </span>
        )}
        {verified && (
          <span
            className="shrink-0 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800"
            title={formatPairingProgress(pairingProgress)}
          >
            verified
          </span>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <EditorToolStrip
          tool={tool}
          editMode={editMode}
          showSegments={showSegments}
          locked={locked}
          hasSelection={hasSelection}
          onPickTool={onPickTool}
          onToggleEdit={onToggleEdit}
          onToggleLines={onToggleLines}
        />

        <EditorWorkflowMenu
          locked={locked}
          binarized={binarized}
          binarizing={binarizing}
          segmenting={segmenting}
          ocrRunning={ocrRunning}
          exporting={exporting}
          hasSegments={hasSegments}
          pdfPanelOpen={pdfPanelOpen}
          pdfPanelMode={pdfPanelMode}
          onBinarize={onBinarize}
          onClearBinarize={onClearBinarize}
          onAutoSegment={onAutoSegment}
          onOcrPage={onOcrPage}
          onPdfOpen={onPdfOpen}
          onPdfClose={onPdfClose}
        />

        <EditorMoreMenu
          locked={locked}
          exporting={exporting}
          ocrRunning={ocrRunning}
          busy={binarizing || segmenting || exporting || ocrRunning}
          showKrakenCeiling={showKrakenCeiling}
          canToggleCeiling={canToggleCeiling}
          showHistory={showHistory}
          onLock={onLock}
          onUnlock={onUnlock}
          onExport={onExport}
          onToggleCeiling={onToggleCeiling}
          onDelete={onDelete}
          onToggleHistory={onToggleHistory}
        />
      </div>
    </header>
  );
}
