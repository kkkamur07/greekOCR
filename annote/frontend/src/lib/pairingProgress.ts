import type { PairingProgress, Segment, TextLine } from "@/types/api";

export function segmentIsPaired(segment: Segment): boolean {
  return segment.text_override != null || segment.paired_text_line_index != null;
}

export function computePairingProgress(
  segments: Segment[],
  textLines: TextLine[],
): PairingProgress {
  const pairedCount = segments.filter(segmentIsPaired).length;
  const usedIndices = new Set(
    segments
      .map((s) => s.paired_text_line_index)
      .filter((index): index is number => index != null),
  );
  const unusedLineCount = textLines.filter((line) => !usedIndices.has(line.index)).length;

  return {
    paired_count: pairedCount,
    unpaired_count: segments.length - pairedCount,
    text_line_count: textLines.length,
    unused_line_count: unusedLineCount,
  };
}

export function isPairingComplete(progress: PairingProgress): boolean {
  return (
    progress.unpaired_count === 0 &&
    (progress.text_line_count === 0 || progress.unused_line_count === 0)
  );
}

export function formatPairingProgress(progress: PairingProgress): string {
  const parts = [
    `${progress.paired_count} paired`,
    `${progress.unpaired_count} unpaired`,
  ];
  if (progress.text_line_count > 0) {
    parts.push(`${progress.unused_line_count} unused line${progress.unused_line_count === 1 ? "" : "s"}`);
  }
  return parts.join(" · ");
}
