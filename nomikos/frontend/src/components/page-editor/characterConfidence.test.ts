import { describe, expect, it } from "vitest";

import {
  characterConfidencesForTranscription,
  confidenceHighlightColor,
  confidenceLabelColor,
  confidenceRunTitle,
  containsJoiningScript,
  formatConfidencePercent,
  groupConfidenceRuns,
  hasDistinctCharacterConfidences,
  type CharacterConfidence,
  type LineTranscriptionWithCharacterConfidence,
} from "./characterConfidence";

const BASE_TRANSCRIPTION: LineTranscriptionWithCharacterConfidence = {
  id: "tx-1",
  transcription_id: "model-1",
  transcription_kind: "model",
  text: "abc",
  confidence: 0.82,
};

describe("characterConfidence", () => {
  it("maps confidence values to highlight and label colors", () => {
    expect(confidenceHighlightColor(0.95)).toBe("#059669");
    expect(confidenceHighlightColor(0.8)).toBe("#d97706");
    expect(confidenceHighlightColor(0.6)).toBe("#d97706");
    expect(confidenceHighlightColor(0.2)).toBe("#dc2626");
    expect(confidenceLabelColor(0.95)).toBe("#059669");
    expect(formatConfidencePercent(0.825)).toBe("82.5%");
  });

  it("uses explicit per-character confidences when aligned with text", () => {
    const transcription: LineTranscriptionWithCharacterConfidence = {
      ...BASE_TRANSCRIPTION,
      character_confidences: [
        { char: "a", confidence: 0.99 },
        { char: "b", confidence: 0.55 },
        { char: "c", confidence: 0.71 },
      ],
    };

    expect(hasDistinctCharacterConfidences(transcription)).toBe(true);
    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: "a", confidence: 0.99 },
      { char: "b", confidence: 0.55 },
      { char: "c", confidence: 0.71 },
    ]);
  });

  it("falls back to line confidence for each character when per-char scores are missing", () => {
    expect(hasDistinctCharacterConfidences(BASE_TRANSCRIPTION)).toBe(false);
    expect(characterConfidencesForTranscription(BASE_TRANSCRIPTION)).toEqual([
      { char: "a", confidence: 0.82 },
      { char: "b", confidence: 0.82 },
      { char: "c", confidence: 0.82 },
    ]);
  });
});

describe("groupConfidenceRuns", () => {
  const scored = (
    text: string,
    ...confidences: number[]
  ): CharacterConfidence[] =>
    Array.from(text, (char, index) => ({
      char,
      confidence: confidences[index] ?? confidences[confidences.length - 1],
    }));

  it("collapses a uniform line into one run", () => {
    expect(
      groupConfidenceRuns(scored("\u0720\u0721\u072A\u071D", 0.96)),
    ).toEqual([
      {
        text: "\u0720\u0721\u072A\u071D",
        confidence: 0.96,
        maxConfidence: 0.96,
      },
    ]);
  });

  it("splits only where the tier changes, not where the score changes", () => {
    const runs = groupConfidenceRuns(scored("abcd", 0.99, 0.95, 0.4, 0.42));

    expect(runs.map((run) => run.text)).toEqual(["ab", "cd"]);
    expect(runs[0]).toEqual({
      text: "ab",
      confidence: 0.95,
      maxConfidence: 0.99,
    });
    expect(runs[1]).toEqual({
      text: "cd",
      confidence: 0.4,
      maxConfidence: 0.42,
    });
  });

  it("keeps a combining mark in the run of its base letter", () => {
    // Scored on its own, the seyame would open a low run of one mark and the
    // dots would render away from the letter they belong to.
    const runs = groupConfidenceRuns(
      scored("\u071D\u0308\u0718", 0.95, 0.2, 0.95),
    );

    expect(runs.map((run) => run.text)).toEqual(["\u071D\u0308", "\u0718"]);
  });

  it("keeps an astral character whole", () => {
    // "".split("") would cut the surrogate pair in half.
    const runs = groupConfidenceRuns([
      { char: "\u{10900}", confidence: 0.9 },
      { char: "a", confidence: 0.9 },
    ]);

    expect(runs).toHaveLength(1);
    expect(runs[0].text).toBe("\u{10900}a");
  });

  it("returns nothing for empty input", () => {
    expect(groupConfidenceRuns([])).toEqual([]);
  });
});

describe("containsJoiningScript", () => {
  it("is true for Syriac and Arabic, false for the other scripts served", () => {
    expect(containsJoiningScript("\u0720\u0721")).toBe(true);
    expect(containsJoiningScript("\u0645\u0631")).toBe(true);
    expect(containsJoiningScript("\u0860")).toBe(true);
    expect(containsJoiningScript("\u1F10\u03BD")).toBe(false);
    expect(containsJoiningScript("\u2C81\u2C93")).toBe(false);
    expect(containsJoiningScript("\u0531\u0561")).toBe(false);
    expect(containsJoiningScript("")).toBe(false);
  });
});

describe("confidenceRunTitle", () => {
  it("names one score for a uniform run and a span of scores otherwise", () => {
    expect(
      confidenceRunTitle({ text: "ab", confidence: 0.96, maxConfidence: 0.96 }),
    ).toBe("96.0% confidence (high)");
    expect(
      confidenceRunTitle({ text: "ab", confidence: 0.92, maxConfidence: 0.99 }),
    ).toBe("92.0% to 99.0% confidence (high)");
  });
});
