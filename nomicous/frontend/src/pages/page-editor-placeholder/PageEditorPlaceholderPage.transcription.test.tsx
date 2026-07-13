import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { toast } from "../../components/ui/toast";
import {
  DOCUMENT,
  flushPageEditorEffects,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

describe("PageEditorPlaceholderPage transcription", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("pairs a selected Segment to imported text lines and updates Pairing progress", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
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
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "line-2",
        part_id: "part-1",
        block_id: null,
        order: 1,
        kind: "polygon",
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPagePairing.mockResolvedValue({
      text_lines: [
        { order: 0, text: "alpha", paired_line_id: null },
        { order: 1, text: "beta", paired_line_id: null },
      ],
      pairing_progress: { paired_lines: 0, total_lines: 2, percent: 0 },
    });
    mockedApi.pairTextLine.mockResolvedValue({
      text_lines: [
        { order: 0, text: "alpha", paired_line_id: null },
        { order: 1, text: "beta", paired_line_id: "line-1" },
      ],
      pairing_progress: { paired_lines: 1, total_lines: 2, percent: 50 },
    });

    renderPageEditor();

    expect(
      await screen.findByText("Pairing progress: 0/2 Lines paired"),
    ).toBeTruthy();
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    fireEvent.click(screen.getByRole("button", { name: /pair text line 2/i }));

    await waitFor(() => {
      expect(mockedApi.pairTextLine).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        { line_id: "line-1", text_line_order: 1 },
      );
    });
    expect(
      await screen.findByText("Pairing progress: 1/2 Lines paired"),
    ).toBeTruthy();
  });

  it("saves typed approved text for the selected Segment and refreshes Pairing progress", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
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
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "line-2",
        part_id: "part-1",
        block_id: null,
        order: 1,
        kind: "polygon",
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.getPagePairing
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 0, total_lines: 2, percent: 0 },
      })
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 2, percent: 50 },
      });
    mockedApi.updateGroundTruthLineText.mockResolvedValue({
      id: "line-tx-1",
      transcription_id: "ground-truth-1",
      transcription_kind: "ground_truth",
      text: "typed approved text",
      confidence: null,
      text_source: "human_edited",
      character_confidences: null,
    });

    renderPageEditor();

    fireEvent.click(await screen.findByLabelText(/^Segment 1/));
    fireEvent.change(
      screen.getByLabelText(/approved text for selected segment/i),
      {
        target: { value: "typed approved text" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "typed approved text" },
      );
    });
    expect(
      await screen.findByText("Pairing progress: 1/2 Lines paired"),
    ).toBeTruthy();
  });

  it("switches to Transcription edit mode and saves Ground truth text for the selected Segment", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
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
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [
          {
            id: "line-tx-1",
            transcription_id: "ground-truth-1",
            transcription_kind: "ground_truth",
            text: "old approved text",
            confidence: null,
            text_source: "human_edited",
            character_confidences: null,
          },
          {
            id: "line-tx-2",
            transcription_id: "model-1",
            transcription_kind: "model",
            text: "model suggestion",
            confidence: 0.91,
            text_source: "model",
            character_confidences: null,
          },
        ],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: "ground-truth-1",
        document_id: "doc-1",
        name: "Ground truth",
        kind: "ground_truth",
        created_by_job_id: null,
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "model-1",
        document_id: "doc-1",
        name: "Kraken run",
        kind: "model",
        created_by_job_id: "job-1",
        created_at: "2026-06-16T10:01:00Z",
      },
    ]);
    mockedApi.getPagePairing
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 1, percent: 100 },
      })
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 1, percent: 100 },
      });

    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /transcription edit/i }),
    );
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    expect(
      screen.getByRole("heading", { name: /transcription edit/i }),
    ).toBeTruthy();

    const textArea = screen.getByLabelText(
      /ground truth text for selected segment/i,
    );
    expect(textArea).toHaveValue("old approved text");
    fireEvent.change(textArea, { target: { value: "corrected ground truth" } });
    const toastSuccess = vi.spyOn(toast, "success");
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "corrected ground truth" },
      );
    });
    await waitFor(() => {
      expect(toastSuccess).toHaveBeenCalledWith("Ground truth text saved");
    });
  });

  it("re-runs OCR on the selected segment from the pairing strip", async () => {
    localStorage.setItem("nomicous_inference_preference", "cloud");
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listInferenceModels.mockResolvedValue([
      {
        id: "model-1",
        name: "kraken-transcribe-default",
        provider: "kraken",
        task: "transcribe",
        artifact_ref: "registry://greek-calamari-v1",
        default_params: {},
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.listTranscriptions
      .mockResolvedValueOnce([
        {
          id: "ground-truth-1",
          document_id: "doc-1",
          name: "Ground truth",
          kind: "ground_truth",
          created_by_job_id: null,
          created_at: "2026-06-16T10:00:00Z",
        },
        {
          id: "model-1",
          document_id: "doc-1",
          name: "Model layer",
          kind: "model",
          created_by_job_id: "job-old",
          created_at: "2026-06-16T10:00:00Z",
        },
      ])
      .mockResolvedValueOnce([
        {
          id: "ground-truth-1",
          document_id: "doc-1",
          name: "Ground truth",
          kind: "ground_truth",
          created_by_job_id: null,
          created_at: "2026-06-16T10:00:00Z",
        },
        {
          id: "model-2",
          document_id: "doc-1",
          name: "Model layer 2",
          kind: "model",
          created_by_job_id: "job-new",
          created_at: "2026-06-16T10:00:00Z",
        },
      ]);
    const lineFixture = {
      id: "line-1",
      part_id: "part-1",
      block_id: null,
      order: 0,
      kind: "polygon" as const,
      points: [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ],
      source: "manual" as const,
      source_metadata: null,
      kraken_ceiling: null,
      manual_geometry: true,
      line_transcriptions: [
        {
          id: "line-tx-model-1",
          transcription_id: "model-1",
          transcription_kind: "model" as const,
          text: "old ocr",
          confidence: 0.8,
          text_source: "model" as const,
          character_confidences: null,
        },
      ],
      created_at: "2026-06-16T10:00:00Z",
    };
    mockedApi.listPartLines
      .mockResolvedValueOnce([lineFixture])
      .mockResolvedValueOnce([
        {
          ...lineFixture,
          line_transcriptions: [
            {
              id: "line-tx-model-2",
              transcription_id: "model-2",
              transcription_kind: "model" as const,
              text: "fresh ocr",
              confidence: 0.92,
              text_source: "model" as const,
              character_confidences: null,
            },
          ],
        },
      ]);
    mockedApi.enqueueTranscribePart.mockResolvedValue({ job_id: "job-ocr-1" });
    mockedApi.getJob.mockResolvedValue({
      id: "job-ocr-1",
      type: "transcribe",
      status: "done",
      payload: {},
      result: {
        transcription_id: "model-2",
        lines: [{ line_id: "line-1", text: "fresh ocr", confidence: 0.92 }],
      },
      error: null,
      document_id: "doc-1",
      document_part_id: "part-1",
      created_at: "2026-06-16T10:00:00Z",
      updated_at: "2026-06-16T10:00:00Z",
      started_at: "2026-06-16T10:00:00Z",
      completed_at: "2026-06-16T10:00:00Z",
    });

    renderPageEditor();
    fireEvent.click(await screen.findByLabelText(/^Segment 1/));
    fireEvent.click(
      screen.getByRole("button", { name: /re-run ocr on segment 1/i }),
    );

    await waitFor(() => {
      expect(mockedApi.enqueueTranscribePart).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        { model_id: "model-1", line_ids: ["line-1"] },
      );
    });
    const jobsButton = await screen.findByRole("button", {
      name: /1 background job finished/i,
    });
    expect(jobsButton).toHaveTextContent("1 job finished");

    await waitFor(() => {
      expect(mockedApi.getJob).toHaveBeenCalledWith("job-ocr-1");
    });
  });

  it("shows model OCR review and saves the selected Segment to Ground truth", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
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
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [
          {
            id: "line-tx-model-1",
            transcription_id: "model-1",
            transcription_kind: "model",
            text: "model suggestion",
            confidence: 0.91,
            text_source: "model",
            character_confidences: null,
          },
        ],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: "ground-truth-1",
        document_id: "doc-1",
        name: "Ground truth",
        kind: "ground_truth",
        created_by_job_id: null,
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "model-1",
        document_id: "doc-1",
        name: "Kraken run",
        kind: "model",
        created_by_job_id: "job-1",
        created_at: "2026-06-16T10:01:00Z",
      },
    ]);
    mockedApi.getPagePairing
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 0, total_lines: 1, percent: 0 },
      })
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 1, percent: 100 },
      });

    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /transcription edit/i }),
    );
    fireEvent.click(screen.getByLabelText(/^Segment 1/));

    expect(screen.getByText("Model output:")).toBeTruthy();
    expect(
      screen.getByLabelText("OCR model output for segment 1"),
    ).toBeTruthy();
    fireEvent.change(
      screen.getByLabelText(/approved text for selected segment/i),
      {
        target: { value: "model suggestion" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "model suggestion" },
      );
    });
  });

  it("ignores stale text source metadata on ground truth", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
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
            character_confidences: [
              { char: "m", confidence: 0.95 },
              { char: "o", confidence: 0.62 },
            ],
          },
        ],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: "ground-truth-1",
        document_id: "doc-1",
        name: "Ground truth",
        kind: "ground_truth",
        created_by_job_id: null,
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "model-1",
        document_id: "doc-1",
        name: "Kraken run",
        kind: "model",
        created_by_job_id: "job-1",
        created_at: "2026-06-16T10:01:00Z",
      },
    ]);

    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /transcription edit/i }),
    );
    fireEvent.click(screen.getByLabelText(/^Segment 1/));

    expect(screen.getByText("Model output:")).toBeTruthy();
    expect(
      screen.getByLabelText("OCR model output for segment 1"),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: /accept/i })).toBeNull();
  });

  it("surfaces Ground truth save API errors and keeps the typed text visible", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
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
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [
          {
            id: "line-tx-1",
            transcription_id: "ground-truth-1",
            transcription_kind: "ground_truth",
            text: "old approved text",
            confidence: null,
            text_source: "human_edited",
            character_confidences: null,
          },
        ],
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.updateGroundTruthLineText.mockRejectedValue(
      new ApiError("Only Ground truth transcriptions can be edited.", 400),
    );

    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /transcription edit/i }),
    );
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    const textArea = screen.getByLabelText(
      /ground truth text for selected segment/i,
    );
    fireEvent.change(textArea, { target: { value: "typed but rejected" } });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    expect(
      await screen.findByText(
        "Only Ground truth transcriptions can be edited.",
      ),
    ).toBeTruthy();
    expect(
      screen.getByLabelText(/ground truth text for selected segment/i),
    ).toHaveValue("typed but rejected");
  });
});
