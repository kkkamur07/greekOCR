/**
 * Doing the same thing twice has to say so twice.
 *
 * These messages are toasts and nothing else - `showSticky` deliberately
 * excludes them - so a toast that does not appear is the whole of the feedback
 * gone. Several of the sentences are constants, which is what made a repeat
 * silent.
 */
import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { toast } from "../ui/toast";
import { PageEditorStatusAlerts } from "./PageEditorStatusAlerts";
import { statusMessage } from "./statusMessage";

const IDLE = {
  submissionRefusal: null,
  saveMessage: null,
  transcriptionSaveMessage: null,
  ocrMessage: null,
  segmentMessage: null,
  mutationError: null,
  pairingError: null,
  layoutError: null,
  lineError: null,
};

describe("PageEditorStatusAlerts", () => {
  beforeEach(() => {
    vi.spyOn(toast, "success").mockImplementation(() => {});
    vi.spyOn(toast, "error").mockImplementation(() => {});
  });

  it("toasts a repeated save every time it is saved", () => {
    const { rerender } = render(
      <PageEditorStatusAlerts
        {...IDLE}
        transcriptionSaveMessage={statusMessage("Ground truth text saved")}
      />,
    );
    expect(toast.success).toHaveBeenCalledTimes(1);

    // The same sentence, from a second save. Keyed on the text, this was a
    // dependency that had not changed, and the researcher heard nothing.
    rerender(
      <PageEditorStatusAlerts
        {...IDLE}
        transcriptionSaveMessage={statusMessage("Ground truth text saved")}
      />,
    );

    expect(toast.success).toHaveBeenCalledTimes(2);
    expect(toast.success).toHaveBeenLastCalledWith("Ground truth text saved");
  });

  it("does not toast again when something unrelated re-renders the page", () => {
    const saved = statusMessage("Layout reset");
    const { rerender } = render(
      <PageEditorStatusAlerts {...IDLE} saveMessage={saved} />,
    );
    expect(toast.success).toHaveBeenCalledTimes(1);

    rerender(<PageEditorStatusAlerts {...IDLE} saveMessage={saved} />);

    expect(toast.success).toHaveBeenCalledTimes(1);
  });

  it("keeps each message on its own toast", () => {
    render(
      <PageEditorStatusAlerts
        {...IDLE}
        saveMessage={statusMessage("Manual geometry saved")}
        transcriptionSaveMessage={statusMessage("Saved to Ground truth")}
      />,
    );

    expect(toast.success).toHaveBeenCalledTimes(2);
    expect(toast.success).toHaveBeenCalledWith("Manual geometry saved");
    expect(toast.success).toHaveBeenCalledWith("Saved to Ground truth");
  });
});
