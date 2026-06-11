"use client";

import { useRef, useState } from "react";
import { EDITOR_SHORTCUTS } from "@/hooks/useEditorShortcuts";
import {
  CeilingIcon,
  ExportIcon,
  HistoryIcon,
  LockIcon,
  MoreIcon,
  TrashIcon,
  UnlockIcon,
} from "./icons";
import { MenuRow, MenuSection } from "./MenuPrimitives";
import { useDismissibleMenu } from "./useDismissibleMenu";

interface EditorMoreMenuProps {
  locked: boolean;
  exporting: boolean;
  ocrRunning: boolean;
  busy: boolean;
  showKrakenCeiling: boolean;
  canToggleCeiling: boolean;
  showHistory: boolean;
  onLock: () => void;
  onUnlock: () => void;
  onExport: () => void;
  onToggleCeiling: () => void;
  onDelete: () => void;
  onToggleHistory: () => void;
}

export default function EditorMoreMenu({
  locked,
  exporting,
  ocrRunning,
  busy,
  showKrakenCeiling,
  canToggleCeiling,
  showHistory,
  onLock,
  onUnlock,
  onExport,
  onToggleCeiling,
  onDelete,
  onToggleHistory,
}: EditorMoreMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useDismissibleMenu(open, () => setOpen(false), rootRef);

  const runAndClose = (fn: () => void) => {
    setOpen(false);
    fn();
  };

  const menuActive = open || showHistory || showKrakenCeiling || locked || exporting;

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="More actions"
        className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm ring-1 ${
          menuActive
            ? "bg-gray-100 text-gray-900 ring-gray-300"
            : "bg-white text-gray-700 ring-gray-200 hover:bg-gray-50"
        }`}
      >
        <MoreIcon className="h-3.5 w-3.5 opacity-70" />
        More
        <span className="text-[10px] text-gray-400" aria-hidden>
          ▾
        </span>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-1 w-64 rounded-lg border border-gray-200 bg-white p-1.5 shadow-xl"
        >
          <MenuSection label="Page">
            {locked ? (
              <MenuRow
                icon={<UnlockIcon />}
                label="Unlock page"
                hint="Resume editing annotation"
                onClick={() => runAndClose(onUnlock)}
              />
            ) : (
              <MenuRow
                icon={<LockIcon />}
                label="Lock page"
                hint="Freeze annotation at current state"
                onClick={() => runAndClose(onLock)}
              />
            )}
            <MenuRow
              icon={<ExportIcon />}
              label={exporting ? "Exporting…" : "Export"}
              hint="Save rectified lines to disk"
              prominent
              disabled={busy || ocrRunning}
              onClick={() => runAndClose(onExport)}
            />
          </MenuSection>

          <hr className="my-1 border-gray-100" />

          <MenuSection label="Tools">
            <MenuRow
              icon={<CeilingIcon />}
              label="Kraken ceiling"
              hint={
                canToggleCeiling
                  ? "Magenta dashed overlay on segment"
                  : "Select a Kraken segment first"
              }
              active={showKrakenCeiling}
              disabled={!canToggleCeiling}
              onClick={() => runAndClose(onToggleCeiling)}
            />
            <MenuRow
              icon={<TrashIcon />}
              label="Delete segment"
              hint={EDITOR_SHORTCUTS.delete}
              destructive
              disabled={locked}
              onClick={() => runAndClose(onDelete)}
            />
          </MenuSection>

          <hr className="my-1 border-gray-100" />

          <MenuRow
            icon={<HistoryIcon />}
            label={showHistory ? "Hide history" : "Annotation history"}
            hint={`${EDITOR_SHORTCUTS.undo} quick restore`}
            active={showHistory}
            onClick={() => runAndClose(onToggleHistory)}
          />
        </div>
      )}
    </div>
  );
}
