import { describe, expect, it } from "vitest";

import type { PublicLineResponse } from "../api/client";
import {
  lineTextForLayer,
  linesForPart,
  publicLinesToRegions,
} from "./publicLayout";

const LINE: PublicLineResponse = {
  id: "line-1",
  part_id: "part-1",
  order: 0,
  points: [
    [10, 10],
    [50, 10],
    [50, 30],
    [10, 30],
  ],
  line_transcriptions: [
    {
      id: "tx-1",
      transcription_id: "ground-truth-1",
      transcription_kind: "ground_truth",
      text: "alpha",
      confidence: null,
    },
  ],
};

describe("publicLayout", () => {
  it("filters and sorts lines for a part", () => {
    const lines = linesForPart(
      [
        LINE,
        { ...LINE, id: "line-2", part_id: "part-2", order: 0 },
        {
          ...LINE,
          id: "line-3",
          part_id: "part-1",
          order: 1,
          points: LINE.points,
        },
      ],
      "part-1",
    );
    expect(lines.map((line) => line.id)).toEqual(["line-1", "line-3"]);
  });

  it("maps public lines to canvas regions", () => {
    const regions = publicLinesToRegions([LINE]);
    expect(regions).toHaveLength(1);
    expect(regions[0]?.id).toBe(1);
    expect(regions[0]?.bbox).toEqual([10, 10, 50, 30]);
    expect(lineTextForLayer(LINE, "ground-truth-1")).toBe("alpha");
  });
});
