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
