import { describe, expect, it } from "vitest";

import type {
  LineResponse,
  TranscriptionLayerResponse,
} from "../../../api/client";
import {
  mergeSavedLine,
  modelLayerIdForPromotion,
  segmentHasGroundTruth,
  segmentIdsWithGroundTruth,
  showsModelSourceReview,
  syncLayoutLinesFromSegments,
  transcriptionForOcrReview,
} from "./utils";

const MODEL_LAYER: TranscriptionLayerResponse = {
  id: "model-1",
  document_id: "doc-1",
  name: "Kraken run",
  kind: "model",
  created_by_job_id: "job-1",
  created_at: "2026-06-16T10:01:00Z",
};

const GROUND_TRUTH_LAYER: TranscriptionLayerResponse = {
  id: "ground-truth-1",
  document_id: "doc-1",
  name: "Ground truth",
  kind: "ground_truth",
  created_by_job_id: null,
  created_at: "2026-06-16T10:00:00Z",
};

const LINE = {
  id: "line-1",
  part_id: "part-1",
  block_id: null,
  order: 0,
  kind: "polygon",
  points: [
    [10, 10],
    [50, 10],
    [50, 30],
    [10, 30],
  ],
  baseline: {
    points: [
      [10, 30],
      [50, 30],
    ],
  },
  mask: {
    points: [
      [10, 10],
      [50, 10],
      [50, 30],
      [10, 30],
    ],
  },
  source: "manual",
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [
    {
      id: "line-tx-ground-1",
      transcription_id: "ground-truth-1",
      transcription_kind: "ground_truth",
      text: "model suggestion",
      confidence: null,
    },
    {
      id: "line-tx-model-1",
      transcription_id: "model-1",
      transcription_kind: "model",
      text: "model suggestion",
      confidence: 0.91,
    },
  ],
  created_at: "2026-06-16T10:00:00Z",
} satisfies LineResponse;

describe("page editor transcription utils", () => {
  it("shows OCR review for model transcriptions", () => {
    expect(showsModelSourceReview(LINE.line_transcriptions[1])).toBe(true);
    expect(showsModelSourceReview(LINE.line_transcriptions[0])).toBe(false);
  });

  it("selects OCR review only for a model transcription layer", () => {
    expect(transcriptionForOcrReview(LINE, GROUND_TRUTH_LAYER)).toBeNull();
    expect(transcriptionForOcrReview(LINE, MODEL_LAYER)).toEqual(
      LINE.line_transcriptions[1],
    );
  });

  it("resolves the model layer id used for ground truth promotion", () => {
    expect(modelLayerIdForPromotion(LINE, MODEL_LAYER)).toBe("model-1");
    expect(modelLayerIdForPromotion(LINE, GROUND_TRUTH_LAYER)).toBe("model-1");
  });

  it("treats segments with saved ground truth text as paired on the canvas", () => {
    expect(segmentHasGroundTruth(LINE)).toBe(true);
    expect(segmentIdsWithGroundTruth([LINE])).toEqual(new Set(["line-1"]));
    expect(
      segmentHasGroundTruth({
        ...LINE,
        id: "line-2",
        line_transcriptions: [],
      }),
    ).toBe(false);
  });

  it("syncs layout line geometry from segment state", () => {
    const layout = syncLayoutLinesFromSegments(
      {
        blocks: [],
        lines: [
          {
            id: "line-1",
            baseline: {
              points: [
                [0, 0],
                [10, 0],
              ],
            },
            manual_geometry: false,
          },
        ],
      },
      [
        {
          ...LINE,
          baseline: {
            points: [
              [60, 140],
              [300, 150],
            ],
          },
          manual_geometry: true,
        },
      ],
    );

    expect(layout.lines[0]?.baseline).toEqual({
      points: [
        [60, 140],
        [300, 150],
      ],
    });
    expect(layout.lines[0]?.manual_geometry).toBe(true);
  });

  it("merges a saved line into segment state by id or appends in order", () => {
    const krakenLine = {
      ...LINE,
      id: "line-2",
      order: 1,
      block_id: "block-1",
      source: "kraken",
      source_metadata: { model: "kraken:blla" },
      kraken_ceiling: [
        [-1, 9],
        [11, 9],
        [11, 16],
        [-1, 16],
      ],
    } as LineResponse;

    expect(
      mergeSavedLine([LINE], {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      }),
    ).toEqual([
      {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      },
    ]);
    expect(mergeSavedLine([LINE], krakenLine)).toEqual([LINE, krakenLine]);
    expect(
      mergeSavedLine([krakenLine, LINE], {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      }),
    ).toEqual([
      krakenLine,
      {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      },
    ]);
  });
});
