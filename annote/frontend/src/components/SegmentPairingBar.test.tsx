import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Segment, TextLine } from "@/types/api";

import SegmentPairingBar from "./SegmentPairingBar";

const textLines: TextLine[] = [
  { index: 1, text: "first line" },
  { index: 2, text: "second line" },
  { index: 3, text: "third line" },
];

const segment: Segment = {
  id: "seg-1",
  number: 1,
  kind: "polygon",
  points: [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
  ],
  paired_text_line_index: 1,
};

const otherSegment: Segment = {
  ...segment,
  id: "seg-2",
  number: 2,
  paired_text_line_index: 2,
};

describe("SegmentPairingBar", () => {
  it("shows paired transcription lines in green and unpaired lines in amber", () => {
    render(
      <SegmentPairingBar
        segment={segment}
        textLines={textLines}
        segments={[segment, otherSegment]}
        onPair={vi.fn()}
        onTextOverride={vi.fn()}
        onClose={vi.fn()}
        onDone={vi.fn()}
      />,
    );

    const activeLine = screen.getByRole("button", { name: /1\.\s*first line/i });
    const pairedElsewhere = screen.getByRole("button", { name: /2\.\s*second line/i });
    const unpairedLine = screen.getByRole("button", { name: /3\.\s*third line/i });

    expect(activeLine.className).toContain("border-green-600");
    expect(pairedElsewhere.className).toContain("border-green-300");
    expect(unpairedLine.className).toContain("border-amber-300");
  });

  it("saves then finishes when Enter is pressed in the transcription field", () => {
    const onSave = vi.fn();
    const onDone = vi.fn();

    render(
      <SegmentPairingBar
        segment={segment}
        textLines={textLines}
        segments={[segment]}
        onPair={vi.fn()}
        onTextOverride={vi.fn()}
        onSave={onSave}
        onClose={vi.fn()}
        onDone={onDone}
      />,
    );

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });

    expect(onSave).toHaveBeenCalledOnce();
    expect(onDone).toHaveBeenCalledOnce();
  });
});
