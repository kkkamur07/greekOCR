import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { api, type DocumentWithPartsResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { PageEditorPlaceholderPage } from './PageEditorPlaceholderPage';

vi.mock('../components/AuthenticatedImage', () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  const mockedApi = {
    ...actual.api,
    getDocument: vi.fn(),
    getPartLayout: vi.fn(),
    listPartLines: vi.fn(),
    listTranscriptions: vi.fn(),
    getPagePairing: vi.fn(),
    importPageTranscription: vi.fn(),
    pairTextLine: vi.fn(),
    updateGroundTruthLineText: vi.fn(),
    copyToGroundTruth: vi.fn(),
    updatePartReviewStatus: vi.fn(),
    replacePartLines: vi.fn(),
    updateLineGeometry: vi.fn(),
    resetPartLayout: vi.fn(),
    generateTranscriptionPdf: vi.fn(),
    exportPageXml: vi.fn(),
    updateDocument: vi.fn(),
    listInferenceModels: vi.fn(),
    resolvePartModelBinding: vi.fn(),
    enqueueTranscribePart: vi.fn(),
    getJob: vi.fn(),
  };

  async function waitForJob(
    jobId: string,
    options?: { timeoutMs?: number; onUpdate?: (job: actual.JobResponse) => void },
  ): Promise<actual.JobResponse> {
    const timeoutMs = options?.timeoutMs ?? 120_000;
    const deadline = Date.now() + timeoutMs;
    let lastJob: actual.JobResponse | null = null;
    while (Date.now() < deadline) {
      const job = await mockedApi.getJob(jobId);
      if (!lastJob || lastJob.status !== job.status || lastJob.updated_at !== job.updated_at) {
        options?.onUpdate?.(job);
      }
      lastJob = job;
      if (job.status === 'done') return job;
      if (job.status === 'failed') {
        throw new Error(job.error ?? 'Job failed.');
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    throw new Error('Job timed out.');
  }

  return {
    ...actual,
    api: mockedApi,
    waitForJob: vi.fn(waitForJob),
  };
});

type MockedEditorApi = {
  getDocument: ReturnType<typeof vi.fn>;
  getPartLayout: ReturnType<typeof vi.fn>;
  listPartLines: ReturnType<typeof vi.fn>;
  listTranscriptions: ReturnType<typeof vi.fn>;
  getPagePairing: ReturnType<typeof vi.fn>;
  importPageTranscription: ReturnType<typeof vi.fn>;
  pairTextLine: ReturnType<typeof vi.fn>;
  updateGroundTruthLineText: ReturnType<typeof vi.fn>;
  copyToGroundTruth: ReturnType<typeof vi.fn>;
  updatePartReviewStatus: ReturnType<typeof vi.fn>;
  replacePartLines: ReturnType<typeof vi.fn>;
  updateLineGeometry: ReturnType<typeof vi.fn>;
  resetPartLayout: ReturnType<typeof vi.fn>;
  generateTranscriptionPdf: ReturnType<typeof vi.fn>;
  exportPageXml: ReturnType<typeof vi.fn>;
  updateDocument: ReturnType<typeof vi.fn>;
  listInferenceModels: ReturnType<typeof vi.fn>;
  resolvePartModelBinding: ReturnType<typeof vi.fn>;
  enqueueTranscribePart: ReturnType<typeof vi.fn>;
  getJob: ReturnType<typeof vi.fn>;
};

const mockedApi = api as unknown as MockedEditorApi;

const DOCUMENT: DocumentWithPartsResponse = {
  id: 'doc-1',
  project_id: 'project-1',
  name: 'Grec 1360',
  workflow: 'draft',
  created_at: '2026-06-16T10:00:00Z',
  updated_at: '2026-06-16T10:00:00Z',
  parts: [
    {
      id: 'part-1',
      document_id: 'doc-1',
      order: 0,
      image_url: '/media/parts/part-1',
      width: 640,
      height: 900,
      reviewed: false,
      created_at: '2026-06-16T10:00:00Z',
    },
  ],
};

function renderPageEditor() {
  return render(
    <MemoryRouter initialEntries={['/projects/project-1/documents/doc-1/parts/part-1']}>
      <Routes>
        <Route
          path="/projects/:projectId/documents/:documentId/parts/:partId"
          element={<PageEditorPlaceholderPage />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

async function enableBaselinesOnCanvas() {
  fireEvent.click(await screen.findByRole('button', { name: /editor settings/i }));
  const checkbox = screen.getByRole('checkbox', { name: /show line baselines/i });
  if (!(checkbox as HTMLInputElement).checked) {
    fireEvent.click(checkbox);
  }
}

describe('PageEditorPlaceholderPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    mockedApi.listPartLines.mockResolvedValue([]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: 'ground-truth-1',
        document_id: 'doc-1',
        name: 'Ground truth',
        kind: 'ground_truth',
        created_by_job_id: null,
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
    mockedApi.listInferenceModels.mockResolvedValue([]);
    mockedApi.resolvePartModelBinding.mockResolvedValue({
      binding: { id: 'binding-1', task: 'transcribe', model_id: 'model-1', overrides: {} },
      model: {
        id: 'model-1',
        name: 'kraken-transcribe-default',
        provider: 'kraken',
        task: 'transcribe',
        artifact_ref: 'x',
        default_params: {},
        created_at: '2026-06-16T10:00:00Z',
      },
      effective_params: {},
    });
    mockedApi.updateDocument.mockResolvedValue({
      ...DOCUMENT,
      workflow: 'published',
    });
    mockedApi.importPageTranscription.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
    mockedApi.pairTextLine.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
    mockedApi.updateGroundTruthLineText.mockResolvedValue({
      id: 'line-tx-1',
      transcription_id: 'ground-truth-1',
      transcription_kind: 'ground_truth',
      text: 'typed approved text',
      confidence: null,
      text_source: 'human_edited',
      character_confidences: null,
    });
    mockedApi.copyToGroundTruth.mockResolvedValue({ copied_line_ids: ['line-1'] });
    mockedApi.updatePartReviewStatus.mockResolvedValue({
      ...DOCUMENT.parts[0],
      reviewed: true,
    });
    mockedApi.replacePartLines.mockImplementation(async (_projectId, _documentId, _partId, body) =>
      body.lines.map((line: Record<string, unknown>, index: number) => ({
        id: (line.id as string | undefined) ?? `line-${index + 1}`,
        part_id: 'part-1',
        block_id: null,
        order: line.order,
        kind: line.kind,
        points: line.points,
        baseline: line.baseline ?? { points: [] },
        mask: line.mask ?? null,
        source: line.source,
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      })),
    );
  });

  afterEach(async () => {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
  });

  it('opens an authenticated document part in the annote page workspace', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    expect(screen.getByText('Grec 1360 · Page 1')).toBeTruthy();
    expect(screen.getByAltText('Page 1')).toBeTruthy();
    expect(screen.getByRole('link', { name: /back to document/i }).getAttribute('href')).toBe(
      '/projects/project-1/documents/doc-1',
    );
  });

  it('renders classic strip layout with toolbar, canvas, and pairing strip when a Segment is selected', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    expect(screen.getByRole('button', { name: /rectangle segment/i })).toBeTruthy();
    expect(screen.getByRole('img', { name: /page geometry canvas/i })).toBeTruthy();

    fireEvent.click(screen.getByLabelText(/^Segment 1/));

    expect(screen.getByLabelText(/transcription layer/i)).toBeTruthy();
    expect(screen.getByText('Segment 1')).toBeTruthy();
    expect(screen.getByRole('progressbar', { name: /pairing progress/i })).toBeTruthy();
  });

  it('does not render protected media when the API rejects access', async () => {
    mockedApi.getDocument.mockRejectedValue(new ApiError('Forbidden', 403));

    renderPageEditor();

    expect(await screen.findByText('Page unavailable')).toBeTruthy();
    expect(screen.getByText('This page is not available to your account.')).toBeTruthy();
    expect(screen.queryByAltText('Page 1')).toBeNull();
  });

  it('renders part layout blocks and line baselines in layout edit mode', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: 'block-1',
        order: 0,
        kind: 'polygon',
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: [
          [60, 140],
          [300, 150],
        ],
        mask: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        source: 'kraken',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [
        {
          id: 'block-1',
          box: [
            [40, 60],
            [320, 60],
            [320, 220],
            [40, 220],
          ],
          manual_geometry: false,
        },
      ],
      lines: [
        {
          id: 'line-1',
          block_id: 'block-1',
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: false,
        },
      ],
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    expect(await screen.findByRole('heading', { name: /layout edit/i })).toBeTruthy();
    expect(screen.getByLabelText('Block block-1')).toBeTruthy();
    expect(screen.getByLabelText('Line line-1 baseline')).toBeTruthy();
  });

  it('renders layout geometry when API returns box and baseline objects', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: 'block-1',
        order: 0,
        kind: 'polygon',
        points: [
          [40, 60],
          [320, 60],
          [320, 220],
          [40, 220],
        ],
        baseline: {
          points: [
            [60, 140],
            [300, 150],
          ],
        },
        mask: null,
        source: 'kraken',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [
        {
          id: 'block-1',
          box: {
            points: [
              [40, 60],
              [320, 60],
              [320, 220],
              [40, 220],
            ],
          },
          manual_geometry: false,
        },
      ],
      lines: [
        {
          id: 'line-1',
          block_id: 'block-1',
          baseline: {
            points: [
              [60, 140],
              [300, 150],
            ],
          },
          manual_geometry: false,
        },
      ],
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    const baseline = await screen.findByLabelText('Line line-1 baseline');
    expect(baseline.getAttribute('points')).toBe('60,140 300,150');
  });

  it('draws a rectangle Segment and saves it as Line geometry for the document part', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /rectangle segment/i }));

    const canvas = screen.getByRole('img', { name: /page geometry canvas/i });
    fireEvent.mouseDown(canvas, { clientX: 20, clientY: 30 });
    fireEvent.mouseMove(canvas, { clientX: 120, clientY: 80 });
    fireEvent.mouseUp(canvas, { clientX: 120, clientY: 80 });

    await waitFor(() => {
      expect(mockedApi.replacePartLines).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        {
          lines: [
            {
              order: 0,
              kind: 'rectangle',
              points: [
                [20, 30],
                [120, 30],
                [120, 80],
                [20, 80],
              ],
              source: 'manual',
            },
          ],
        },
      );
    });
    expect(await screen.findByText('1 Segment')).toBeTruthy();
  });

  it('pairs a selected Segment to imported text lines and updates Pairing progress', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
      {
        id: 'line-2',
        part_id: 'part-1',
        block_id: null,
        order: 1,
        kind: 'polygon',
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPagePairing.mockResolvedValue({
      text_lines: [
        { order: 0, text: 'alpha', paired_line_id: null },
        { order: 1, text: 'beta', paired_line_id: null },
      ],
      pairing_progress: { paired_lines: 0, total_lines: 2, percent: 0 },
    });
    mockedApi.pairTextLine.mockResolvedValue({
      text_lines: [
        { order: 0, text: 'alpha', paired_line_id: null },
        { order: 1, text: 'beta', paired_line_id: 'line-1' },
      ],
      pairing_progress: { paired_lines: 1, total_lines: 2, percent: 50 },
    });

    renderPageEditor();

    expect(await screen.findByText('Pairing progress: 0/2 Lines paired')).toBeTruthy();
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    fireEvent.click(screen.getByRole('button', { name: /pair text line 2/i }));

    await waitFor(() => {
      expect(mockedApi.pairTextLine).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { line_id: 'line-1', text_line_order: 1 },
      );
    });
    expect(await screen.findByText('Pairing progress: 1/2 Lines paired')).toBeTruthy();
  });

  it('saves typed approved text for the selected Segment and refreshes Pairing progress', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
      {
        id: 'line-2',
        part_id: 'part-1',
        block_id: null,
        order: 1,
        kind: 'polygon',
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
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
      id: 'line-tx-1',
      transcription_id: 'ground-truth-1',
      transcription_kind: 'ground_truth',
      text: 'typed approved text',
      confidence: null,
      text_source: 'human_edited',
      character_confidences: null,
    });

    renderPageEditor();

    fireEvent.click(await screen.findByLabelText(/^Segment 1/));
    fireEvent.change(screen.getByLabelText(/approved text for selected segment/i), {
      target: { value: 'typed approved text' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'ground-truth-1',
        'line-1',
        { text: 'typed approved text' },
      );
    });
    expect(await screen.findByText('Pairing progress: 1/2 Lines paired')).toBeTruthy();
  });

  it('switches to Transcription edit mode and saves Ground truth text for the selected Segment', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [
          {
            id: 'line-tx-1',
            transcription_id: 'ground-truth-1',
            transcription_kind: 'ground_truth',
            text: 'old approved text',
            confidence: null,
      text_source: 'human_edited',
      character_confidences: null,
          },
          {
            id: 'line-tx-2',
            transcription_id: 'model-1',
            transcription_kind: 'model',
            text: 'model suggestion',
            confidence: 0.91,
      text_source: 'model',
      character_confidences: null,
          },
        ],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: 'ground-truth-1',
        document_id: 'doc-1',
        name: 'Ground truth',
        kind: 'ground_truth',
        created_by_job_id: null,
        created_at: '2026-06-16T10:00:00Z',
      },
      {
        id: 'model-1',
        document_id: 'doc-1',
        name: 'Kraken run',
        kind: 'model',
        created_by_job_id: 'job-1',
        created_at: '2026-06-16T10:01:00Z',
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

    fireEvent.click(await screen.findByRole('button', { name: /transcription edit/i }));
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    expect(screen.getByRole('heading', { name: /transcription edit/i })).toBeTruthy();
    expect(screen.getByLabelText(/transcription layer/i)).toHaveValue('ground-truth-1');

    const textArea = screen.getByLabelText(/ground truth text for selected segment/i);
    expect(textArea).toHaveValue('old approved text');
    fireEvent.change(textArea, { target: { value: 'corrected ground truth' } });
    fireEvent.click(screen.getByRole('button', { name: /save ground truth/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'ground-truth-1',
        'line-1',
        { text: 'corrected ground truth' },
      );
    });
    expect(await screen.findByText('Ground truth text saved')).toBeTruthy();
  });

  it('re-runs OCR on the selected segment from the pairing strip', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listInferenceModels.mockResolvedValue([
      {
        id: 'model-1',
        name: 'kraken-transcribe-default',
        provider: 'kraken',
        task: 'transcribe',
        artifact_ref: 'x',
        default_params: {},
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.listTranscriptions
      .mockResolvedValueOnce([
        {
          id: 'ground-truth-1',
          document_id: 'doc-1',
          name: 'Ground truth',
          kind: 'ground_truth',
          created_by_job_id: null,
          created_at: '2026-06-16T10:00:00Z',
        },
        {
          id: 'model-1',
          document_id: 'doc-1',
          name: 'Model layer',
          kind: 'model',
          created_by_job_id: 'job-old',
          created_at: '2026-06-16T10:00:00Z',
        },
      ])
      .mockResolvedValueOnce([
        {
          id: 'ground-truth-1',
          document_id: 'doc-1',
          name: 'Ground truth',
          kind: 'ground_truth',
          created_by_job_id: null,
          created_at: '2026-06-16T10:00:00Z',
        },
        {
          id: 'model-2',
          document_id: 'doc-1',
          name: 'Model layer 2',
          kind: 'model',
          created_by_job_id: 'job-new',
          created_at: '2026-06-16T10:00:00Z',
        },
      ]);
    const lineFixture = {
      id: 'line-1',
      part_id: 'part-1',
      block_id: null,
      order: 0,
      kind: 'polygon' as const,
      points: [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ],
      source: 'manual' as const,
      source_metadata: null,
      kraken_ceiling: null,
      manual_geometry: true,
      line_transcriptions: [
        {
          id: 'line-tx-model-1',
          transcription_id: 'model-1',
          transcription_kind: 'model' as const,
          text: 'old ocr',
          confidence: 0.8,
          text_source: 'model' as const,
          character_confidences: null,
        },
      ],
      created_at: '2026-06-16T10:00:00Z',
    };
    mockedApi.listPartLines
      .mockResolvedValueOnce([lineFixture])
      .mockResolvedValueOnce([
        {
          ...lineFixture,
          line_transcriptions: [
            {
              id: 'line-tx-model-2',
              transcription_id: 'model-2',
              transcription_kind: 'model' as const,
              text: 'fresh ocr',
              confidence: 0.92,
              text_source: 'model' as const,
              character_confidences: null,
            },
          ],
        },
      ]);
    mockedApi.enqueueTranscribePart.mockResolvedValue({ job_id: 'job-ocr-1' });
    mockedApi.getJob.mockResolvedValue({
      id: 'job-ocr-1',
      type: 'transcribe',
      status: 'done',
      payload: {},
      result: {
        transcription_id: 'model-2',
        lines: [{ line_id: 'line-1', text: 'fresh ocr', confidence: 0.92 }],
      },
      error: null,
      document_id: 'doc-1',
      document_part_id: 'part-1',
      created_at: '2026-06-16T10:00:00Z',
      updated_at: '2026-06-16T10:00:00Z',
      started_at: '2026-06-16T10:00:00Z',
      completed_at: '2026-06-16T10:00:00Z',
    });

    renderPageEditor();
    fireEvent.click(await screen.findByLabelText(/^Segment 1/));
    fireEvent.click(screen.getByRole('button', { name: /re-run ocr on segment 1/i }));

    await waitFor(() => {
      expect(mockedApi.enqueueTranscribePart).toHaveBeenCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { model_id: 'model-1', line_ids: ['line-1'] },
      );
    });
    const dialog = await screen.findByRole('dialog', { name: /background jobs/i });
    expect(within(dialog).getByText('Segment 1')).toBeTruthy();
    expect(within(dialog).getByText('Segment OCR')).toBeTruthy();

    await waitFor(() => {
      expect(mockedApi.getJob).toHaveBeenCalledWith('job-ocr-1');
    });
  });

  it('shows model OCR review and saves the selected Segment to Ground truth', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines
      .mockResolvedValueOnce([
        {
          id: 'line-1',
          part_id: 'part-1',
          block_id: null,
          order: 0,
          kind: 'polygon',
          points: [
            [10, 10],
            [50, 10],
            [50, 30],
            [10, 30],
          ],
          source: 'manual',
          source_metadata: null,
          kraken_ceiling: null,
          manual_geometry: true,
          line_transcriptions: [
            {
              id: 'line-tx-model-1',
              transcription_id: 'model-1',
              transcription_kind: 'model',
              text: 'model suggestion',
              confidence: 0.91,
              text_source: 'model',
              character_confidences: null,
            },
          ],
          created_at: '2026-06-16T10:00:00Z',
        },
      ])
      .mockResolvedValueOnce([
        {
          id: 'line-1',
          part_id: 'part-1',
          block_id: null,
          order: 0,
          kind: 'polygon',
          points: [
            [10, 10],
            [50, 10],
            [50, 30],
            [10, 30],
          ],
          source: 'manual',
          source_metadata: null,
          kraken_ceiling: null,
          manual_geometry: true,
          line_transcriptions: [
            {
              id: 'line-tx-ground-1',
              transcription_id: 'ground-truth-1',
              transcription_kind: 'ground_truth',
              text: 'model suggestion',
              confidence: null,
      text_source: 'human_edited',
      character_confidences: null,
            },
            {
              id: 'line-tx-model-1',
              transcription_id: 'model-1',
              transcription_kind: 'model',
              text: 'model suggestion',
              confidence: 0.91,
              text_source: 'model',
              character_confidences: null,
            },
          ],
          created_at: '2026-06-16T10:00:00Z',
        },
      ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: 'ground-truth-1',
        document_id: 'doc-1',
        name: 'Ground truth',
        kind: 'ground_truth',
        created_by_job_id: null,
        created_at: '2026-06-16T10:00:00Z',
      },
      {
        id: 'model-1',
        document_id: 'doc-1',
        name: 'Kraken run',
        kind: 'model',
        created_by_job_id: 'job-1',
        created_at: '2026-06-16T10:01:00Z',
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

    fireEvent.click(await screen.findByRole('button', { name: /transcription edit/i }));
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    fireEvent.change(screen.getByLabelText(/transcription layer/i), {
      target: { value: 'model-1' },
    });

    expect(screen.getByText('Model output:')).toBeTruthy();
    expect(screen.getByLabelText('OCR model output for segment 1')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /accept/i }));

    await waitFor(() => {
      expect(mockedApi.copyToGroundTruth).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'model-1',
        { line_ids: ['line-1'] },
      );
    });
    expect(await screen.findByText('Saved to Ground truth')).toBeTruthy();
    expect(screen.getByLabelText(/transcription layer/i)).toHaveValue('ground-truth-1');
    expect(screen.getByLabelText(/ground truth text for selected segment/i)).toHaveValue(
      'model suggestion',
    );
  });

  it('shows OCR review on Ground truth when text_source is model', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [[10, 10], [50, 10], [50, 30], [10, 30]],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [
          {
            id: 'line-tx-ground-1',
            transcription_id: 'ground-truth-1',
            transcription_kind: 'ground_truth',
            text: 'model suggestion',
            confidence: null,
            text_source: 'model',
          },
          {
            id: 'line-tx-model-1',
            transcription_id: 'model-1',
            transcription_kind: 'model',
            text: 'model suggestion',
            confidence: 0.91,
            text_source: 'model',
            character_confidences: [
              { char: 'm', confidence: 0.95 },
              { char: 'o', confidence: 0.62 },
            ],
          },
        ],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: 'ground-truth-1',
        document_id: 'doc-1',
        name: 'Ground truth',
        kind: 'ground_truth',
        created_by_job_id: null,
        created_at: '2026-06-16T10:00:00Z',
      },
      {
        id: 'model-1',
        document_id: 'doc-1',
        name: 'Kraken run',
        kind: 'model',
        created_by_job_id: 'job-1',
        created_at: '2026-06-16T10:01:00Z',
      },
    ]);

    renderPageEditor();

    fireEvent.click(await screen.findByRole('button', { name: /transcription edit/i }));
    fireEvent.click(screen.getByLabelText(/^Segment 1/));

    expect(screen.getByText('Model output:')).toBeTruthy();
    expect(screen.getByLabelText('OCR model output for segment 1')).toBeTruthy();
    expect(screen.queryByLabelText(/ground truth text for selected segment/i)).toBeNull();
    expect(screen.getByRole('button', { name: /accept/i })).toBeTruthy();
  });

  it('surfaces Ground truth save API errors and keeps the typed text visible', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [
          {
            id: 'line-tx-1',
            transcription_id: 'ground-truth-1',
            transcription_kind: 'ground_truth',
            text: 'old approved text',
            confidence: null,
      text_source: 'human_edited',
      character_confidences: null,
          },
        ],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.updateGroundTruthLineText.mockRejectedValue(
      new ApiError('Only Ground truth transcriptions can be edited.', 400),
    );

    renderPageEditor();

    fireEvent.click(await screen.findByRole('button', { name: /transcription edit/i }));
    fireEvent.click(screen.getByLabelText(/^Segment 1/));
    const textArea = screen.getByLabelText(/ground truth text for selected segment/i);
    fireEvent.change(textArea, { target: { value: 'typed but rejected' } });
    fireEvent.click(screen.getByRole('button', { name: /save ground truth/i }));

    expect(
      await screen.findByText('Only Ground truth transcriptions can be edited.'),
    ).toBeTruthy();
    expect(screen.getByLabelText(/ground truth text for selected segment/i)).toHaveValue(
      'typed but rejected',
    );
  });

  it('draws a polygon Segment and saves it as Line geometry for the document part', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /polygon segment/i }));

    const canvas = screen.getByRole('img', { name: /page geometry canvas/i });
    fireEvent.click(canvas, { clientX: 40, clientY: 40 });
    fireEvent.click(canvas, { clientX: 160, clientY: 45 });
    fireEvent.click(canvas, { clientX: 150, clientY: 90 });
    fireEvent.click(canvas, { clientX: 35, clientY: 85 });
    fireEvent.doubleClick(canvas);

    await waitFor(() => {
      expect(mockedApi.replacePartLines).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        {
          lines: [
            {
              order: 0,
              kind: 'polygon',
              points: [
                [40, 40],
                [160, 45],
                [150, 90],
                [35, 85],
              ],
              source: 'manual',
            },
          ],
        },
      );
    });
    expect(await screen.findByText('1 Segment')).toBeTruthy();
  });

  it('deletes a selected Segment and saves the remaining Line geometry', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'rectangle',
        points: [
          [10, 10],
          [50, 10],
          [50, 30],
          [10, 30],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
      {
        id: 'line-2',
        part_id: 'part-1',
        block_id: null,
        order: 1,
        kind: 'polygon',
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
        source: 'manual',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);

    renderPageEditor();

    fireEvent.click(await screen.findByLabelText(/^Segment 1/));
    fireEvent.click(screen.getByRole('button', { name: /delete segment/i }));

    await waitFor(() => {
      expect(mockedApi.replacePartLines).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        {
          lines: [
            {
              id: 'line-2',
              order: 0,
              kind: 'polygon',
              points: [
                [80, 20],
                [120, 20],
                [120, 50],
                [80, 50],
              ],
              source: 'manual',
            },
          ],
        },
      );
    });
  });

  it('edits a Line baseline and saves it as manual geometry', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: {
          points: [
            [60, 140],
            [300, 150],
          ],
        },
        mask: {
          points: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
        },
        source: 'kraken',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: 'line-1',
          baseline: {
            points: [
              [60, 140],
              [300, 150],
            ],
          },
          mask: {
            points: [
              [55, 110],
              [305, 118],
              [300, 178],
              [50, 168],
            ],
          },
          manual_geometry: false,
        },
      ],
    });
    mockedApi.updateLineGeometry.mockResolvedValue({
      id: 'line-1',
      baseline: [
        [60, 145],
        [300, 155],
      ],
      manual_geometry: true,
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText('Line line-1 baseline'));
    fireEvent.click(screen.getByRole('button', { name: /move baseline down/i }));
    fireEvent.click(screen.getByRole('button', { name: /save layout/i }));

    await waitFor(() => {
      expect(mockedApi.updateLineGeometry).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        'line-1',
        {
          baseline: {
            points: [
              [60, 145],
              [300, 155],
            ],
          },
          mask: {
            points: [
              [55, 110],
              [305, 118],
              [300, 178],
              [50, 168],
            ],
          },
        },
      );
    });
    expect(await screen.findByText('Manual geometry saved')).toBeTruthy();
  });

  it('resets selected Line layout through the API and refreshes the canvas state', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: [
          [60, 140],
          [300, 150],
        ],
        mask: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        source: 'kraken',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: 'line-1',
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: true,
        },
      ],
    });
    mockedApi.resetPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: 'line-1',
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: false,
        },
      ],
    });

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText('Line line-1 baseline'));
    fireEvent.click(screen.getByRole('button', { name: /reset layout/i }));

    await waitFor(() => {
      expect(mockedApi.resetPartLayout).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { line_ids: ['line-1'] },
      );
    });
    expect(await screen.findByText('Layout reset')).toBeTruthy();
  });

  it('shows a member-only error when the layout save API rejects access', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      {
        id: 'line-1',
        part_id: 'part-1',
        block_id: null,
        order: 0,
        kind: 'polygon',
        points: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        baseline: [
          [60, 140],
          [300, 150],
        ],
        mask: [
          [55, 110],
          [305, 118],
          [300, 178],
          [50, 168],
        ],
        source: 'kraken',
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: false,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      },
    ]);
    mockedApi.getPartLayout.mockResolvedValue({
      blocks: [],
      lines: [
        {
          id: 'line-1',
          baseline: [
            [60, 140],
            [300, 150],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
          manual_geometry: false,
        },
      ],
    });
    mockedApi.updateLineGeometry.mockRejectedValue(new ApiError('Forbidden', 403));

    renderPageEditor();

    await enableBaselinesOnCanvas();

    fireEvent.click(await screen.findByLabelText('Line line-1 baseline'));
    fireEvent.click(screen.getByRole('button', { name: /move baseline down/i }));
    fireEvent.click(screen.getByRole('button', { name: /save layout/i }));

    expect(
      await screen.findByText('Only project members can edit layout.'),
    ).toBeTruthy();
    expect(screen.getByLabelText('Line line-1 baseline').getAttribute('points')).toBe(
      '60,140 300,150',
    );
  });

  it('opens the transcription PDF side pane from the toolbar', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.generateTranscriptionPdf.mockResolvedValue(
      new Blob(['%PDF'], { type: 'application/pdf' }),
    );
    const originalUrl = globalThis.URL;
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:preview'),
      revokeObjectURL: vi.fn(),
    });

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /toggle transcription pdf/i }));

    expect(await screen.findByLabelText('Transcription PDF preview')).toBeTruthy();
    await waitFor(() => {
      expect(mockedApi.generateTranscriptionPdf).toHaveBeenCalledWith(
        'project-1',
        'doc-1',
        'part-1',
      );
    });

    vi.stubGlobal('URL', originalUrl);
  });

  it('publishes the document from the Process sharing section', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    fireEvent.click(await screen.findByRole('button', { name: /^process/i }));
    fireEvent.click(await screen.findByRole('menuitem', { name: /publish live page/i }));

    await waitFor(() => {
      expect(mockedApi.updateDocument).toHaveBeenCalledWith('project-1', 'doc-1', {
        workflow: 'published',
      });
    });
    expect(screen.getByLabelText(/public document url/i)).toBeTruthy();
  });
});
