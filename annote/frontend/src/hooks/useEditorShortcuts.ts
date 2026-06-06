"use client";

import { useEffect } from "react";
import type { DrawTool } from "@/types/api";

export const EDITOR_SHORTCUTS = {
  pan: "H",
  select: "V",
  polygon: "P",
  rectangle: "R",
  editVertices: "E",
  toggleLines: "L",
  undoLastPoint: "⌘Z",
  delete: "Del",
  fitPage: "0",
  zoomIn: "=",
  zoomOut: "-",
} as const;

interface EditorShortcutHandlers {
  onTool: (tool: DrawTool) => void;
  onToggleEdit: () => void;
  onToggleLines: () => void;
  onUndoLastPoint: () => void;
  onDelete: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitPage: () => void;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable;
}

export function useEditorShortcuts(handlers: EditorShortcutHandlers): void {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;

      const key = e.key;
      if (key === " " || e.code === "Space") return;

      if (key === "h" || key === "H") {
        e.preventDefault();
        handlers.onTool("pan");
        return;
      }
      if (key === "v" || key === "V") {
        e.preventDefault();
        handlers.onTool("select");
        return;
      }
      if (key === "p" || key === "P") {
        e.preventDefault();
        handlers.onTool("polygon");
        return;
      }
      if (key === "r" || key === "R") {
        e.preventDefault();
        handlers.onTool("rectangle");
        return;
      }
      if (key === "e" || key === "E") {
        e.preventDefault();
        handlers.onToggleEdit();
        return;
      }
      if (key === "l" || key === "L") {
        e.preventDefault();
        handlers.onToggleLines();
        return;
      }
      if (key === "0") {
        e.preventDefault();
        handlers.onFitPage();
        return;
      }
      if (key === "=" || key === "+") {
        e.preventDefault();
        handlers.onZoomIn();
        return;
      }
      if (key === "-") {
        e.preventDefault();
        handlers.onZoomOut();
        return;
      }
      if ((key === "z" || key === "Z") && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        e.preventDefault();
        handlers.onUndoLastPoint();
        return;
      }
      if (key === "Delete" || (key === "Backspace" && !e.metaKey && !e.ctrlKey)) {
        e.preventDefault();
        handlers.onDelete();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}
