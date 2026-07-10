import { useEffect } from 'react';

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
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input/textarea
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const key = e.key.toLowerCase();
      const ctrl = e.ctrlKey || e.metaKey;

      // B - Draw Box
      if (key === 'b' && !ctrl) {
        e.preventDefault();
        handlers.onDrawBox?.();
      }
      // P - Draw Polygon
      else if (key === 'p' && !ctrl) {
        e.preventDefault();
        handlers.onDrawPolygon?.();
      }
      // V - Edit Vertices
      else if (key === 'v' && !ctrl) {
        e.preventDefault();
        handlers.onEditVertices?.();
      }
      // Delete/Backspace - Delete selected
      else if ((key === 'delete' || key === 'backspace') && !ctrl) {
        e.preventDefault();
        handlers.onDelete?.();
      }
      // Escape - Cancel
      else if (key === 'escape') {
        e.preventDefault();
        handlers.onEscape?.();
      }
      // Enter - Complete (e.g. polygon)
      else if (key === 'enter') {
        handlers.onEnter?.();
      }
      // Ctrl+Z - Undo
      else if (ctrl && key === 'z' && !e.shiftKey) {
        e.preventDefault();
        handlers.onUndo?.();
      }
      // Ctrl+Shift+Z or Ctrl+Y - Redo
      else if ((ctrl && e.shiftKey && key === 'z') || (ctrl && key === 'y')) {
        e.preventDefault();
        handlers.onRedo?.();
      }
      // Arrow keys - Move selected region
      else if (key === 'arrowup') {
        e.preventDefault();
        handlers.onMoveUp?.();
      }
      else if (key === 'arrowdown') {
        e.preventDefault();
        handlers.onMoveDown?.();
      }
      else if (key === 'arrowleft') {
        e.preventDefault();
        handlers.onMoveLeft?.();
      }
      else if (key === 'arrowright') {
        e.preventDefault();
        handlers.onMoveRight?.();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlers]);
};
