import { useEffect, useRef } from "react";

interface ShortcutHandlers {
  onDrawBox?: () => void;
  onDrawPolygon?: () => void;
  onEditVertices?: () => void;
  onDelete?: () => void;
  onEscape?: () => void;
  onEnter?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onMoveLeft?: () => void;
  onMoveRight?: () => void;
}

export const useKeyboardShortcuts = (handlers: ShortcutHandlers) => {
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input/textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const current = handlersRef.current;
      const key = e.key.toLowerCase();
      const ctrl = e.ctrlKey || e.metaKey;

      // B - Draw Box
      if (key === "b" && !ctrl) {
        e.preventDefault();
        current.onDrawBox?.();
      }
      // P - Draw Polygon
      else if (key === "p" && !ctrl) {
        e.preventDefault();
        current.onDrawPolygon?.();
      }
      // V - Edit Vertices
      else if (key === "v" && !ctrl) {
        e.preventDefault();
        current.onEditVertices?.();
      }
      // Delete/Backspace - Delete selected
      else if ((key === "delete" || key === "backspace") && !ctrl) {
        e.preventDefault();
        current.onDelete?.();
      }
      // Escape - Cancel
      else if (key === "escape") {
        e.preventDefault();
        current.onEscape?.();
      }
      // Enter - Complete (e.g. polygon)
      else if (key === "enter") {
        current.onEnter?.();
      }
      // Ctrl+Z - Undo
      else if (ctrl && key === "z" && !e.shiftKey) {
        e.preventDefault();
        current.onUndo?.();
      }
      // Ctrl+Shift+Z or Ctrl+Y - Redo
      else if ((ctrl && e.shiftKey && key === "z") || (ctrl && key === "y")) {
        e.preventDefault();
        current.onRedo?.();
      }
      // Arrow keys - Move selected region
      else if (key === "arrowup") {
        e.preventDefault();
        current.onMoveUp?.();
      } else if (key === "arrowdown") {
        e.preventDefault();
        current.onMoveDown?.();
      } else if (key === "arrowleft") {
        e.preventDefault();
        current.onMoveLeft?.();
      } else if (key === "arrowright") {
        e.preventDefault();
        current.onMoveRight?.();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);
};
