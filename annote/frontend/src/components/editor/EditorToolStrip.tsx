"use client";

import type { ReactNode } from "react";
import type { DrawTool } from "@/types/api";
import { EDITOR_SHORTCUTS } from "@/hooks/useEditorShortcuts";
import { EditIcon, LinesIcon, PanIcon, PolyIcon, RectIcon, SelectIcon } from "./icons";

interface EditorToolStripProps {
  tool: DrawTool;
  editMode: boolean;
  showSegments: boolean;
  locked: boolean;
  hasSelection: boolean;
  onPickTool: (tool: DrawTool) => void;
  onToggleEdit: () => void;
  onToggleLines: () => void;
}

function toolChip(active: boolean) {
  return active
    ? "bg-gray-900 text-white"
    : "text-gray-700 hover:bg-white";
}

interface ToolButtonProps {
  label: string;
  shortcut: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}

function ToolButton({ label, shortcut, active, disabled, onClick, children }: ToolButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex h-9 w-9 flex-col items-center justify-center rounded-md disabled:cursor-not-allowed disabled:opacity-40 ${toolChip(active)}`}
      title={`${label} (${shortcut})`}
      aria-label={`${label} (${shortcut})`}
    >
      {children}
      <kbd className="mt-0.5 font-mono text-[9px] leading-none opacity-70">{shortcut}</kbd>
    </button>
  );
}

export default function EditorToolStrip({
  tool,
  editMode,
  showSegments,
  locked,
  hasSelection,
  onPickTool,
  onToggleEdit,
  onToggleLines,
}: EditorToolStripProps) {
  return (
    <div className="flex items-center gap-0.5 rounded-lg bg-gray-100 p-0.5 ring-1 ring-gray-200/80">
      <ToolButton
        label="Pan"
        shortcut={EDITOR_SHORTCUTS.pan}
        active={tool === "pan"}
        onClick={() => onPickTool("pan")}
      >
        <PanIcon />
      </ToolButton>
      <ToolButton
        label="Select"
        shortcut={EDITOR_SHORTCUTS.select}
        active={tool === "select" && !editMode}
        disabled={locked}
        onClick={() => onPickTool("select")}
      >
        <SelectIcon />
      </ToolButton>
      <ToolButton
        label="Rectangle"
        shortcut={EDITOR_SHORTCUTS.rectangle}
        active={tool === "rectangle"}
        disabled={locked}
        onClick={() => onPickTool("rectangle")}
      >
        <RectIcon />
      </ToolButton>
      <ToolButton
        label="Polygon"
        shortcut={EDITOR_SHORTCUTS.polygon}
        active={tool === "polygon"}
        disabled={locked}
        onClick={() => onPickTool("polygon")}
      >
        <PolyIcon />
      </ToolButton>
      <ToolButton
        label="Edit vertices"
        shortcut={EDITOR_SHORTCUTS.editVertices}
        active={editMode}
        disabled={!hasSelection || locked}
        onClick={onToggleEdit}
      >
        <EditIcon />
      </ToolButton>
      <span className="mx-0.5 h-5 w-px bg-gray-300" aria-hidden />
      <ToolButton
        label="Show lines"
        shortcut={EDITOR_SHORTCUTS.toggleLines}
        active={showSegments}
        onClick={onToggleLines}
      >
        <LinesIcon />
      </ToolButton>
    </div>
  );
}
