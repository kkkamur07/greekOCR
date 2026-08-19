import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useFileDrop } from "./useFileDrop";

function fireDrag(
  type: "dragenter" | "dragover" | "dragleave" | "drop",
  { types = ["Files"], files = [] as File[] } = {},
) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", {
    value: { types, files },
  });
  act(() => {
    window.dispatchEvent(event);
  });
  return event;
}

describe("useFileDrop", () => {
  it("activates while a file drag is over the window and hands over the drop", () => {
    const onFiles = vi.fn();
    const { result } = renderHook(() => useFileDrop(onFiles, true));
    expect(result.current).toBe(false);

    fireDrag("dragenter");
    expect(result.current).toBe(true);

    const file = new File(["x"], "page-1.jpg", { type: "image/jpeg" });
    fireDrag("drop", { files: [file] });
    expect(result.current).toBe(false);
    expect(onFiles).toHaveBeenCalledWith([file]);
  });

  it("only deactivates when the drag leaves the outermost element", () => {
    const { result } = renderHook(() => useFileDrop(vi.fn(), true));

    // Crossing into a child fires another enter before the parent's leave.
    fireDrag("dragenter");
    fireDrag("dragenter");
    fireDrag("dragleave");
    expect(result.current).toBe(true);

    fireDrag("dragleave");
    expect(result.current).toBe(false);
  });

  it("prevents the browser's open-the-file default only for file drags", () => {
    renderHook(() => useFileDrop(vi.fn(), true));
    expect(fireDrag("dragover").defaultPrevented).toBe(true);
    expect(
      fireDrag("dragover", { types: ["text/plain"] }).defaultPrevented,
    ).toBe(false);
  });

  it("ignores drags that carry no files", () => {
    const onFiles = vi.fn();
    const { result } = renderHook(() => useFileDrop(onFiles, true));

    fireDrag("dragenter", { types: ["text/plain"] });
    expect(result.current).toBe(false);
    fireDrag("drop", { types: ["text/plain"] });
    expect(onFiles).not.toHaveBeenCalled();
  });

  it("does nothing while disabled, and resets if disabled mid-drag", () => {
    const onFiles = vi.fn();
    const { result, rerender } = renderHook(
      ({ enabled }) => useFileDrop(onFiles, enabled),
      { initialProps: { enabled: true } },
    );

    fireDrag("dragenter");
    expect(result.current).toBe(true);

    rerender({ enabled: false });
    expect(result.current).toBe(false);
    fireDrag("drop", { files: [new File(["x"], "a.jpg")] });
    expect(onFiles).not.toHaveBeenCalled();
  });
});
