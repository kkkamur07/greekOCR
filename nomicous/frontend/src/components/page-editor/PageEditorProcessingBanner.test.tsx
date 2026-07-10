import { describe, expect, it } from "vitest";

import { getPageEditorProcessingLabel } from "./PageEditorProcessingBanner";

describe("getPageEditorProcessingLabel", () => {
  it("returns null when idle", () => {
    expect(getPageEditorProcessingLabel(null)).toBeNull();
  });

  it("shows transcription labels by scope", () => {
    expect(getPageEditorProcessingLabel("segmentation")).toBe(
      "Segmentation in progress",
    );
    expect(getPageEditorProcessingLabel("transcription-page")).toBe(
      "Full-page transcription in progress",
    );
    expect(getPageEditorProcessingLabel("transcription-segment")).toBe(
      "Segment transcription in progress",
    );
  });
});
