import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi } from 'vitest';

import { api, type DocumentWithPartsResponse } from '../../api/client';
import { BackgroundJobsProvider } from '../../context/BackgroundJobsContext';
import { BackgroundJobsPanel } from '../../components/BackgroundJobsPanel';
import { PageEditorPlaceholderPage } from '../PageEditorPlaceholderPage';

vi.mock('../../auth/session', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../auth/session')>();
  return {
    ...actual,
    hasAccessToken: vi.fn(() => true),
    redirectToLogin: vi.fn(),
  };
});

vi.mock('../../components/AuthenticatedImage', () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
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
    createPartLine: vi.fn(),
    patchPartLine: vi.fn(),
    deletePartLine: vi.fn(),
    updateLineGeometry: vi.fn(),
    resetPartLayout: vi.fn(),
    generateTranscriptionPdf: vi.fn(),
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

export type MockedEditorApi = {
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
  createPartLine: ReturnType<typeof vi.fn>;
  patchPartLine: ReturnType<typeof vi.fn>;
  deletePartLine: ReturnType<typeof vi.fn>;
  updateLineGeometry: ReturnType<typeof vi.fn>;
  resetPartLayout: ReturnType<typeof vi.fn>;
  generateTranscriptionPdf: ReturnType<typeof vi.fn>;
  updateDocument: ReturnType<typeof vi.fn>;
  listInferenceModels: ReturnType<typeof vi.fn>;
  resolvePartModelBinding: ReturnType<typeof vi.fn>;
  enqueueTranscribePart: ReturnType<typeof vi.fn>;
  getJob: ReturnType<typeof vi.fn>;
};

export const mockedApi = api as unknown as MockedEditorApi;

export const DOCUMENT: DocumentWithPartsResponse = {
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

export function renderPageEditor() {
  return render(
    <MemoryRouter initialEntries={['/projects/project-1/documents/doc-1/parts/part-1']}>
      <BackgroundJobsProvider>
        <Routes>
          <Route
            path="/projects/:projectId/documents/:documentId/parts/:partId"
            element={<PageEditorPlaceholderPage />}
          />
        </Routes>
        <BackgroundJobsPanel />
      </BackgroundJobsProvider>
    </MemoryRouter>,
  );
}

export async function enableBaselinesOnCanvas() {
  fireEvent.click(await screen.findByRole('button', { name: /editor settings/i }));
  const checkbox = screen.getByRole('checkbox', { name: /show line baselines/i });
  if (!(checkbox as HTMLInputElement).checked) {
    fireEvent.click(checkbox);
  }
}

export async function flushPageEditorEffects() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
}

export function resetPageEditorApiMocks() {
  vi.clearAllMocks();
  mockedApi.getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
  mockedApi.listPartLines.mockReset();
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
  mockedApi.deletePartLine.mockResolvedValue(undefined);
  mockedApi.createPartLine.mockImplementation(async (_projectId, _documentId, _partId, body) => ({
    id: `line-${body.order + 1}`,
    part_id: 'part-1',
    block_id: body.block_id ?? null,
    order: body.order,
    kind: body.kind,
    points: body.points,
    baseline: body.baseline ?? { points: body.points },
    mask: body.mask ?? { points: body.points },
    source: 'manual',
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: true,
    line_transcriptions: [],
    created_at: '2026-06-16T10:00:00Z',
  }));
  mockedApi.patchPartLine.mockImplementation(
    async (_projectId, _documentId, _partId, lineId, body) => {
      const existing =
        (await mockedApi.listPartLines.mock.results.at(-1)?.value as
          | Array<Record<string, unknown>>
          | undefined)?.find((line) => line.id === lineId) ?? {
          id: lineId,
          part_id: 'part-1',
          block_id: null,
          order: 0,
          kind: 'polygon',
          points: body.points ?? [],
          baseline: { points: [] },
          mask: null,
          source: 'manual',
          source_metadata: null,
          kraken_ceiling: null,
          manual_geometry: true,
          line_transcriptions: [],
          created_at: '2026-06-16T10:00:00Z',
        };
      return {
        ...existing,
        ...body,
        manual_geometry: true,
        source: 'manual',
      };
    },
  );
  mockedApi.replacePartLines.mockImplementation(async (_projectId, _documentId, _partId, body) =>
    body.lines.map((line: Record<string, unknown>, index: number) => ({
      id: (line.id as string | undefined) ?? `line-${index + 1}`,
      part_id: 'part-1',
      block_id: line.block_id ?? null,
      order: line.order,
      kind: line.kind,
      points: line.points,
      baseline: line.baseline ?? { points: [] },
      mask: line.mask ?? null,
      source: line.source,
      source_metadata: line.source_metadata ?? null,
      kraken_ceiling: line.kraken_ceiling ?? null,
      manual_geometry: true,
      line_transcriptions: [],
      created_at: '2026-06-16T10:00:00Z',
    })),
  );
}
