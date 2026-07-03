import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api, type DocumentWithPartsResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { PageEditorPlaceholderPage } from './PageEditorPlaceholderPage';

vi.mock('../components/AuthenticatedImage', () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    api: {
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
    },
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
    });
    mockedApi.copyToGroundTruth.mockResolvedValue({ copied_line_ids: ['line-1'] });
    mockedApi.updatePartReviewStatus.mockResolvedValue({
      ...DOCUMENT.parts[0],
      reviewed: true,
    });
    mockedApi.replacePartLines.mockImplementation(async (_projectId, _documentId, _partId, body) =>
      body.lines.map((line: Record<string, unknown>, index: number) => ({
        id: `line-${index + 1}`,
        part_id: 'part-1',
        block_id: null,
        order: line.order,
        kind: line.kind,
        points: line.points,
        source: line.source,
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: '2026-06-16T10:00:00Z',
      })),
    );
  });

  it('opens an authenticated document part in the annote page workspace', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText('ANNOTE PAGE WORKSPACE')).toBeTruthy();
    expect(screen.getByText('Grec 1360 · Page 1')).toBeTruthy();
    expect(screen.getByAltText('Page 1')).toBeTruthy();
    expect(screen.getByRole('link', { name: /document parts/i }).getAttribute('href')).toBe(
      '/projects/project-1/documents/doc-1',
    );
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

    expect(await screen.findByRole('heading', { name: /layout edit/i })).toBeTruthy();
    expect(screen.getByLabelText('Block block-1')).toBeTruthy();
    expect(screen.getByLabelText('Line line-1 baseline')).toBeTruthy();
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

  it('imports candidate Text lines, pairs the selected Segment, and updates Pairing progress', async () => {
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
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 2, percent: 0 },
    });
    mockedApi.importPageTranscription.mockResolvedValue({
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
    fireEvent.change(screen.getByLabelText(/page transcription text/i), {
      target: { value: 'alpha\n\nbeta' },
    });
    fireEvent.click(screen.getByRole('button', { name: /import page transcription/i }));

    expect(await screen.findByText('Text line 2')).toBeTruthy();
    fireEvent.click(screen.getByLabelText('Segment 1'));
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
    expect(screen.getByText('Text line 2 · paired with Segment 1')).toBeTruthy();
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
    });

    renderPageEditor();

    fireEvent.click(await screen.findByLabelText('Segment 1'));
    fireEvent.change(screen.getByLabelText(/approved text for selected segment/i), {
      target: { value: 'typed approved text' },
    });
    fireEvent.click(screen.getByRole('button', { name: /save approved text/i }));

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
          },
          {
            id: 'line-tx-2',
            transcription_id: 'model-1',
            transcription_kind: 'model',
            text: 'model suggestion',
            confidence: 0.91,
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
    fireEvent.click(screen.getByLabelText('Segment 1'));
    expect(screen.getByRole('heading', { name: /transcription edit/i })).toBeTruthy();
    expect(screen.getByLabelText(/transcription layer/i)).toHaveValue('ground-truth-1');

    const textArea = screen.getByLabelText(/ground truth text for selected segment/i);
    expect(textArea).toHaveValue('old approved text');
    fireEvent.change(textArea, { target: { value: 'corrected ground truth' } });
    fireEvent.click(screen.getByRole('button', { name: /save ground truth text/i }));

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

  it('shows model layer text as read-only and copies the selected Segment to Ground truth', async () => {
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
            },
            {
              id: 'line-tx-model-1',
              transcription_id: 'model-1',
              transcription_kind: 'model',
              text: 'model suggestion',
              confidence: 0.91,
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
    fireEvent.click(screen.getByLabelText('Segment 1'));
    fireEvent.change(screen.getByLabelText(/transcription layer/i), {
      target: { value: 'model-1' },
    });

    expect(screen.getByLabelText(/read-only text for selected segment/i)).toHaveValue(
      'model suggestion',
    );
    expect(screen.getByLabelText(/read-only text for selected segment/i)).toHaveAttribute(
      'readonly',
    );
    fireEvent.click(screen.getByRole('button', { name: /copy selected segment to ground truth/i }));

    await waitFor(() => {
      expect(mockedApi.copyToGroundTruth).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'model-1',
        { line_ids: ['line-1'] },
      );
    });
    expect(await screen.findByText('Copied 1 Segment to Ground truth')).toBeTruthy();
    fireEvent.change(screen.getByLabelText(/transcription layer/i), {
      target: { value: 'ground-truth-1' },
    });
    expect(screen.getByLabelText(/ground truth text for selected segment/i)).toHaveValue(
      'model suggestion',
    );
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
    fireEvent.click(screen.getByLabelText('Segment 1'));
    const textArea = screen.getByLabelText(/ground truth text for selected segment/i);
    fireEvent.change(textArea, { target: { value: 'typed but rejected' } });
    fireEvent.click(screen.getByRole('button', { name: /save ground truth text/i }));

    expect(
      await screen.findByText('Only Ground truth transcriptions can be edited.'),
    ).toBeTruthy();
    expect(screen.getByLabelText(/ground truth text for selected segment/i)).toHaveValue(
      'typed but rejected',
    );
  });

  it('shows Review status and lets a project member mark the Page reviewed', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 1, total_lines: 2, percent: 50 },
    });

    renderPageEditor();

    expect(await screen.findByText('Review status: Unreviewed')).toBeTruthy();
    expect(screen.getByText('Pairing progress: 1/2 Lines paired')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /mark reviewed/i }));

    await waitFor(() => {
      expect(mockedApi.updatePartReviewStatus).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { reviewed: true },
      );
    });
    expect(await screen.findByText('Review status: Reviewed')).toBeTruthy();
    expect(screen.getByText('Pairing progress: 1/2 Lines paired')).toBeTruthy();
  });

  it('loads a Reviewed Page and lets a project member mark it unreviewed', async () => {
    mockedApi.getDocument.mockResolvedValue({
      ...DOCUMENT,
      parts: [{ ...DOCUMENT.parts[0], reviewed: true }],
    });
    mockedApi.updatePartReviewStatus.mockResolvedValue({
      ...DOCUMENT.parts[0],
      reviewed: false,
    });

    renderPageEditor();

    expect(await screen.findByText('Review status: Reviewed')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /mark unreviewed/i }));

    await waitFor(() => {
      expect(mockedApi.updatePartReviewStatus).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        { reviewed: false },
      );
    });
    expect(await screen.findByText('Review status: Unreviewed')).toBeTruthy();
  });

  it('keeps the visible Review status when the API rejects the change', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.updatePartReviewStatus.mockRejectedValue(new ApiError('Forbidden', 403));

    renderPageEditor();

    expect(await screen.findByText('Review status: Unreviewed')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /mark reviewed/i }));

    expect(
      await screen.findByText('Only project members can change Review status.'),
    ).toBeTruthy();
    expect(screen.getByText('Review status: Unreviewed')).toBeTruthy();
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

  it('edits an existing Segment geometry and saves the updated Line points', async () => {
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

    fireEvent.click(await screen.findByLabelText('Segment 1'));
    fireEvent.click(screen.getByRole('button', { name: /move segment right/i }));

    await waitFor(() => {
      expect(mockedApi.replacePartLines).toHaveBeenLastCalledWith(
        'project-1',
        'doc-1',
        'part-1',
        {
          lines: [
            {
              id: 'line-1',
              order: 0,
              kind: 'polygon',
              points: [
                [15, 10],
                [55, 10],
                [55, 30],
                [15, 30],
              ],
              source: 'manual',
            },
          ],
        },
      );
    });
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

    fireEvent.click(await screen.findByLabelText('Segment 1'));
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
    mockedApi.updateLineGeometry.mockResolvedValue({
      id: 'line-1',
      baseline: [
        [60, 145],
        [300, 155],
      ],
      manual_geometry: true,
    });

    renderPageEditor();

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
          baseline: [
            [60, 145],
            [300, 155],
          ],
          mask: [
            [55, 110],
            [305, 118],
            [300, 178],
            [50, 168],
          ],
        },
      );
    });
    expect(await screen.findByText('Manual geometry saved')).toBeTruthy();
  });

  it('resets selected Line layout through the API and refreshes the canvas state', async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
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
});
