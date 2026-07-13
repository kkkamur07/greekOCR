import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PageEditorTranscriptionStrip } from "./PageEditorTranscriptionStrip";

const groundTruthLayer = {
  id: "gt-1",
  document_id: "doc-1",
  name: "Ground truth",
  kind: "ground_truth" as const,
  model_id: null,
  created_at: "2026-06-16T10:00:00Z",
};

const selectedSegment = {
  id: "line-1",
  part_id: "part-1",
  block_id: null,
  order: 0,
  kind: "polygon" as const,
  points: [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
  ] as [number, number][],
  source: "manual" as const,
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [
    {
      id: "tx-1",
      transcription_id: "gt-1",
      transcription_kind: "ground_truth",
      text: "αβ",
      confidence: null,
    },
  ],
  created_at: "2026-06-16T10:00:00Z",
};

describe("PageEditorTranscriptionStrip", () => {
  it("saves Ground truth with a Save control (not 'Save ground truth')", () => {
    const onSaveGroundTruthText = vi.fn();
    render(
      <PageEditorTranscriptionStrip
        visible
        transcriptionLayers={[groundTruthLayer]}
        selectedTranscriptionLayerId="gt-1"
        onSelectTranscriptionLayer={() => undefined}
        selectedSegmentNumber={1}
        selectedSegment={selectedSegment}
        selectedTranscriptionLayer={groundTruthLayer}
        approvedTextDraft="αβ"
        onApprovedTextDraftChange={() => undefined}
        onSaveGroundTruthText={onSaveGroundTruthText}
        onSaveApprovedText={() => undefined}
        onPromoteSelectedSegmentToGroundTruth={() => undefined}
        onRunSegmentOcr={() => undefined}
        onNavigateSegment={() => undefined}
        onDismiss={() => undefined}
        lines={[selectedSegment]}
        selectedSegmentId="line-1"
        transcribeModels={[]}
        selectedTranscribeModelId={null}
        onSelectedTranscribeModelIdChange={() => undefined}
        ocrRunning={false}
      />,
    );

    const save = screen.getByRole("button", { name: /^save$/i });
    expect(save).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: /save ground truth/i }),
    ).toBeNull();
    fireEvent.click(save);
    expect(onSaveGroundTruthText).toHaveBeenCalled();
  });
});
