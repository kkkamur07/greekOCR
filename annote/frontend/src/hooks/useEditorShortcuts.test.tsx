import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useEditorShortcuts, type EditorShortcutHandlers } from "./useEditorShortcuts";

function ShortcutHarness({ handlers }: { handlers: EditorShortcutHandlers }) {
  useEditorShortcuts(handlers);
  return null;
}

function makeHandlers(): EditorShortcutHandlers {
  return {
    onTool: vi.fn(),
    onToggleEdit: vi.fn(),
    onToggleLines: vi.fn(),
    onUndo: vi.fn(),
    onDelete: vi.fn(),
    onSave: vi.fn(),
    onZoomIn: vi.fn(),
    onZoomOut: vi.fn(),
    onFitPage: vi.fn(),
  };
}

describe("useEditorShortcuts", () => {
  it("calls onSave when Enter is pressed outside a text field", () => {
    const handlers = makeHandlers();
    render(<ShortcutHarness handlers={handlers} />);

    fireEvent.keyDown(window, { key: "Enter" });

    expect(handlers.onSave).toHaveBeenCalledOnce();
  });

  it("does not call onSave when Enter is pressed inside a textarea", () => {
    const handlers = makeHandlers();
    render(
      <>
        <ShortcutHarness handlers={handlers} />
        <textarea data-testid="field" />
      </>,
    );

    const field = document.querySelector("[data-testid=field]")!;
    fireEvent.keyDown(field, { key: "Enter" });

    expect(handlers.onSave).not.toHaveBeenCalled();
  });

  it("calls onUndo for Ctrl+Z", () => {
    const handlers = makeHandlers();
    render(<ShortcutHarness handlers={handlers} />);

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });

    expect(handlers.onUndo).toHaveBeenCalledOnce();
  });
});
