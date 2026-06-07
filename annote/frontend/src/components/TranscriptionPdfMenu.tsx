"use client";

import { useEffect, useRef, useState } from "react";
import type { TranscriptionPdfMode } from "@/components/TranscriptionPdfPanel";

interface TranscriptionPdfMenuProps {
  locked: boolean;
  panelOpen: boolean;
  panelMode: TranscriptionPdfMode | null;
  onOpen: (mode: TranscriptionPdfMode) => void;
  onClose: () => void;
}

export default function TranscriptionPdfMenu({
  locked,
  panelOpen,
  panelMode,
  onOpen,
  onClose,
}: TranscriptionPdfMenuProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  const handlePrimaryClick = () => {
    if (panelOpen && panelMode === "preview") {
      onClose();
      return;
    }
    onOpen("preview");
  };

  const selectMode = (mode: TranscriptionPdfMode) => {
    setMenuOpen(false);
    if (mode === "share" && !locked) return;
    if (panelOpen && panelMode === mode) {
      onClose();
      return;
    }
    onOpen(mode);
  };

  return (
    <div ref={rootRef} className="relative flex items-center">
      <button
        type="button"
        onClick={handlePrimaryClick}
        className={`rounded-l px-2.5 py-1 text-sm ${
          panelOpen
            ? "bg-violet-100 text-violet-900"
            : "text-violet-800 hover:bg-violet-50"
        }`}
        title="Open live transcription PDF beside the manuscript"
      >
        Transcription PDF
      </button>
      <button
        type="button"
        onClick={() => setMenuOpen((v) => !v)}
        className={`rounded-r border-l border-violet-200 px-1.5 py-1 text-sm ${
          panelOpen || menuOpen
            ? "bg-violet-100 text-violet-900"
            : "text-violet-800 hover:bg-violet-50"
        }`}
        aria-expanded={menuOpen}
        aria-haspopup="menu"
        title="Preview or share transcription PDF"
      >
        ▾
      </button>

      {menuOpen && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-1 min-w-[11rem] rounded-md border border-gray-200 bg-white py-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => selectMode("preview")}
            className="block w-full px-3 py-1.5 text-left text-sm text-gray-800 hover:bg-gray-50"
          >
            Preview
            <span className="mt-0.5 block text-[11px] text-gray-500">Live blank-page layout beside manuscript</span>
          </button>
          <button
            type="button"
            role="menuitem"
            disabled={!locked}
            onClick={() => selectMode("share")}
            className="block w-full px-3 py-1.5 text-left text-sm hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-400"
          >
            Share
            <span className="mt-0.5 block text-[11px] text-gray-500">
              {locked ? "Frozen PDF from lock time" : "Available when page is locked"}
            </span>
          </button>
        </div>
      )}
    </div>
  );
}
