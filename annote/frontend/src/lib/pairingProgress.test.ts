import { describe, expect, it } from "vitest";

import type { Segment, TextLine } from "@/types/api";

import { computePairingProgress, formatPairingProgress, isPairingComplete } from "./pairingProgress";

const lines: TextLine[] = [
  { index: 1, text: "one" },
  { index: 2, text: "two" },
  { index: 3, text: "three" },
];

const paired: Segment = {
  id: "a",
  number: 1,
  kind: "rectangle",
  points: [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ],
  paired_text_line_index: 1,
};

const unpaired: Segment = {
  ...paired,
  id: "b",
  number: 2,
  paired_text_line_index: null,
};

describe("computePairingProgress", () => {
  it("counts paired segments and unused transcription lines", () => {
    expect(computePairingProgress([paired, unpaired], lines)).toEqual({
      paired_count: 1,
      unpaired_count: 1,
      text_line_count: 3,
      unused_line_count: 2,
    });
  });

  it("treats a text override as paired even without a line index", () => {
    const overrideOnly: Segment = {
      ...unpaired,
      text_override: "typed directly",
    };

    expect(computePairingProgress([overrideOnly], [])).toEqual({
      paired_count: 1,
      unpaired_count: 0,
      text_line_count: 0,
      unused_line_count: 0,
    });
  });
});

describe("formatPairingProgress", () => {
  it("describes paired, unpaired, and unused line counts", () => {
    expect(
      formatPairingProgress({
        paired_count: 2,
        unpaired_count: 1,
        text_line_count: 3,
        unused_line_count: 1,
      }),
    ).toBe("2 paired · 1 unpaired · 1 unused line");
  });
});

describe("isPairingComplete", () => {
  it("is true when every segment is paired and every line is used", () => {
    expect(
      isPairingComplete({
        paired_count: 2,
        unpaired_count: 0,
        text_line_count: 2,
        unused_line_count: 0,
      }),
    ).toBe(true);
  });

  it("is false while segments or transcription lines remain unmatched", () => {
    expect(
      isPairingComplete({
        paired_count: 1,
        unpaired_count: 1,
        text_line_count: 2,
        unused_line_count: 0,
      }),
    ).toBe(false);
  });
});
