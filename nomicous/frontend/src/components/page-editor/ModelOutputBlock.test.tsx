import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ModelOutputBlock } from "./ModelOutputBlock";

describe("ModelOutputBlock", () => {
  it("renders confidence-colored OCR text for a model transcription", () => {
    render(
      <ModelOutputBlock
        segmentNumber={2}
        transcription={{
          id: "tx-1",
          transcription_id: "model-1",
          transcription_kind: "model",
          text: "abc",
          confidence: 0.91,
          character_confidences: [
            { char: "a", confidence: 0.95 },
            { char: "b", confidence: 0.62 },
            { char: "c", confidence: 0.48 },
          ],
        }}
      />,
    );

    expect(screen.getByText("Model output:")).toBeTruthy();
    expect(
      screen.getByLabelText("OCR model output for segment 2"),
    ).toBeTruthy();
    expect(document.querySelector(".ch-high")).toBeTruthy();
    expect(document.querySelector(".ch-mid")).toBeTruthy();
    expect(document.querySelector(".ch-low")).toBeTruthy();
    expect(screen.getByTitle("95.0% confidence (high)")).toBeTruthy();
    expect(screen.getByTitle("62.0% confidence (mid)")).toBeTruthy();
  });

  it("shows a placeholder when no transcription is available", () => {
    render(<ModelOutputBlock segmentNumber={1} transcription={null} />);

    expect(
      screen.getByText("No OCR yet. Run OCR on this segment."),
    ).toBeTruthy();
  });

  it("shows guidance when OCR ran but returned empty text", () => {
    render(
      <ModelOutputBlock
        segmentNumber={1}
        transcription={{
          id: "tx-empty",
          transcription_id: "model-1",
          transcription_kind: "model",
          text: "",
          confidence: 0,
          character_confidences: null,
        }}
      />,
    );

    expect(screen.getByText(/OCR finished with no text/)).toBeTruthy();
  });
});
