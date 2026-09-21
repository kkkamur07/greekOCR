import type {
  CharacterConfidence,
  LineTranscriptionResponse,
} from "../../api/client";

export type { CharacterConfidence };

/**
 * A transcription plus per-character scores. The platform API has no such
 * field (`LineTranscriptionResponse` carries one confidence for the whole
 * line), so this is a client-side extension, declared once, here.
 *
 * Nothing populates it today: a local run's `character_confidences` are sent
 * to the server by `persistLocalTranscribe` and are not returned, so every
 * transcription the editor holds falls back to the per-line confidence below.
 */
export type LineTranscriptionWithCharacterConfidence =
  LineTranscriptionResponse & {
    character_confidences?: CharacterConfidence[] | null;
  };

export function confidenceTierClass(confidence: number): string {
  if (confidence > 0.9) return "ch-high";
  if (confidence > 0.5) return "ch-mid";
  return "ch-low";
}

/** Human-readable tier for aria / tooltips (matches strip legend). */
export function confidenceTierLabel(confidence: number): string {
  if (confidence > 0.9) return "high";
  if (confidence > 0.5) return "mid";
  return "low";
}

export function confidenceHighlightColor(confidence: number): string {
  if (confidence > 0.9) return "#059669";
  if (confidence > 0.7) return "#d97706";
  if (confidence > 0.5) return "#d97706";
  return "#dc2626";
}

export function confidenceLabelColor(confidence: number): string {
  if (confidence > 0.9) return "#059669";
  if (confidence > 0.7) return "#d97706";
  if (confidence > 0.5) return "#d97706";
  return "#dc2626";
}

export function formatConfidencePercent(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function characterConfidencesForTranscription(
  transcription: LineTranscriptionWithCharacterConfidence,
): CharacterConfidence[] {
  const explicit = transcription.character_confidences;
  if (explicit && explicit.length === transcription.text.length) {
    return explicit;
  }
  if (transcription.confidence === null) {
    return transcription.text
      .split("")
      .map((char) => ({ char, confidence: 0 }));
  }
  return transcription.text.split("").map((char) => ({
    char,
    confidence: transcription.confidence as number,
  }));
}

export function hasDistinctCharacterConfidences(
  transcription: LineTranscriptionWithCharacterConfidence,
): boolean {
  const explicit = transcription.character_confidences;
  return Boolean(explicit && explicit.length === transcription.text.length);
}

/**
 * Unicode blocks whose letters are cursive and joining: the font picks an
 * initial, medial, final or isolated glyph from the neighbours, and a browser
 * only shapes across characters that sit in the same inline box with the same
 * layout. Syriac and Arabic and their supplements are the ones this platform
 * transcribes.
 */
const JOINING_SCRIPT_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x0600, 0x06ff], // Arabic
  [0x0700, 0x074f], // Syriac
  [0x0750, 0x077f], // Arabic Supplement
  [0x0860, 0x086f], // Syriac Supplement
  [0x0870, 0x089f], // Arabic Extended-B
  [0x08a0, 0x08ff], // Arabic Extended-A
  [0xfb50, 0xfdff], // Arabic Presentation Forms-A
  [0xfe70, 0xfeff], // Arabic Presentation Forms-B
];

export function containsJoiningScript(text: string): boolean {
  for (const char of text) {
    const code = char.codePointAt(0);
    if (code === undefined) continue;
    for (const [start, end] of JOINING_SCRIPT_RANGES) {
      if (code >= start && code <= end) return true;
    }
  }
  return false;
}

/**
 * Grapheme clusters, so a base letter keeps its combining marks (Syriac vowel
 * points and seyame U+0308, the Coptic supralinear stroke) and an astral
 * character keeps its surrogate pair. `Array.from` is the code point fallback
 * for a runtime without `Intl.Segmenter`.
 */
function graphemeClusters(text: string): string[] {
  const segmenter = (
    Intl as typeof Intl & { Segmenter?: typeof Intl.Segmenter }
  ).Segmenter;
  if (segmenter) {
    return Array.from(
      new segmenter(undefined, { granularity: "grapheme" }).segment(text),
      (entry) => entry.segment,
    );
  }
  return Array.from(text);
}

/** One rendered span: consecutive grapheme clusters sharing a confidence tier. */
export type ConfidenceRun = {
  text: string;
  /** Lowest score in the run. Drives the tier, so a weak mark is never hidden. */
  confidence: number;
  /** Highest score in the run, for the tooltip when the run is not uniform. */
  maxConfidence: number;
};

/**
 * Collapse per-character scores into the fewest spans that still show every
 * tier. One span per character breaks a cursive script: the browser shapes
 * each box on its own, so every Syriac letter falls back to its isolated form
 * and the word reads as loose letters. Grouping also keeps a combining mark in
 * the same box as its base letter.
 */
export function groupConfidenceRuns(
  characterConfidences: CharacterConfidence[],
): ConfidenceRun[] {
  const text = characterConfidences.map((entry) => entry.char).join("");
  const runs: ConfidenceRun[] = [];
  let entryIndex = 0;
  let consumedUnits = 0;

  for (const cluster of graphemeClusters(text)) {
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    let remaining = cluster.length;

    while (remaining > 0 && entryIndex < characterConfidences.length) {
      const entry = characterConfidences[entryIndex];
      const available = entry.char.length - consumedUnits;
      const taken = Math.min(available, remaining);
      lowest = Math.min(lowest, entry.confidence);
      highest = Math.max(highest, entry.confidence);
      remaining -= taken;
      consumedUnits += taken;
      if (consumedUnits >= entry.char.length) {
        entryIndex += 1;
        consumedUnits = 0;
      }
    }

    const confidence = Number.isFinite(lowest) ? lowest : 0;
    const maxConfidence = Number.isFinite(highest) ? highest : confidence;
    const previous = runs[runs.length - 1];
    if (
      previous &&
      confidenceTierClass(previous.confidence) ===
        confidenceTierClass(confidence)
    ) {
      previous.text += cluster;
      previous.confidence = Math.min(previous.confidence, confidence);
      previous.maxConfidence = Math.max(previous.maxConfidence, maxConfidence);
    } else {
      runs.push({ text: cluster, confidence, maxConfidence });
    }
  }

  return runs;
}

/** Tooltip for a run: one score, or the span of scores it covers. */
export function confidenceRunTitle(run: ConfidenceRun): string {
  const tier = confidenceTierLabel(run.confidence);
  const lowest = formatConfidencePercent(run.confidence);
  if (
    Math.round(run.confidence * 1000) === Math.round(run.maxConfidence * 1000)
  ) {
    return `${lowest} confidence (${tier})`;
  }
  return `${lowest} to ${formatConfidencePercent(run.maxConfidence)} confidence (${tier})`;
}
