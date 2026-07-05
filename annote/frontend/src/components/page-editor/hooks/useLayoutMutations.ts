import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import {
  api,
  waitForJob,
  type JobResponse,
  type LayoutLineResponse,
  type LinePoint,
  type LineResponse,
  type LineUpsertRequest,
  type LinesReplaceRequest,
  type PartLayoutResponse,
} from '../../../api/client';
import { ApiError } from '../../../api/errors';
import { cleanPolygonPoints, offsetGeometry } from '../canvasGeometry';
import type { PageEditorJobKind } from '../jobProgress';
import { upsertLineRequest, applyLayoutLineGeometryToSegments, syncLayoutLinesFromSegments } from './utils';

function layoutMutationMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return 'Only project members can edit layout.';
  }
  return error instanceof Error ? error.message : 'Layout update failed.';
}

type LayoutMutationsInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  layout: PartLayoutResponse;
  setLayout: Dispatch<SetStateAction<PartLayoutResponse>>;
  lines: LineResponse[];
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  setLineError: Dispatch<SetStateAction<string | null>>;
  setTextLines: Dispatch<
    SetStateAction<{ order: number; text: string; paired_line_id: string | null }[]>
  >;
  setPairingProgress: Dispatch<
    SetStateAction<{ paired_lines: number; total_lines: number; percent: number }>
  >;
  setPairingError: Dispatch<SetStateAction<string | null>>;
  selectedSegmentId: string | null;
  setSelectedSegmentId: Dispatch<SetStateAction<string | null>>;
  setApprovedTextDraft: Dispatch<SetStateAction<string>>;
  onDrawComplete: () => void;
  trackJobAndWait?: (
    jobId: string,
    meta: { label: string; kind: PageEditorJobKind },
  ) => Promise<JobResponse>;
};

export function useLayoutMutations({
  projectId,
  documentId,
  partId,
  layout,
  setLayout,
  lines,
  setLines,
  setLineError,
  setTextLines,
  setPairingProgress,
  setPairingError,
  selectedSegmentId,
  setSelectedSegmentId,
  setApprovedTextDraft,
  onDrawComplete,
  trackJobAndWait,
}: LayoutMutationsInput) {
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [selectedLineSnapshot, setSelectedLineSnapshot] = useState<{
    baseline?: LayoutLineResponse['baseline'];
    mask?: LayoutLineResponse['mask'];
  } | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const [segmenting, setSegmenting] = useState(false);
  const [useOtsuRefinement, setUseOtsuRefinement] = useState(false);
  const [otsuSphereRadius, setOtsuSphereRadius] = useState(4);
  const [segmentMessage, setSegmentMessage] = useState<string | null>(null);

  useEffect(() => {
    setSelectedLineId(null);
    setSelectedLineSnapshot(null);
    setSaveMessage(null);
    setMutationError(null);
    setSegmentMessage(null);
  }, [projectId, documentId, partId]);

  const sortedLines = [...lines].sort((a, b) => a.order - b.order);

  function moveSelectedBaseline(deltaY: number) {
    if (!selectedLineId) return;
    setSaveMessage(null);
    setMutationError(null);
    setLayout((current) => {
      const nextLayoutLines = current.lines.map((line) =>
        line.id === selectedLineId
          ? {
              ...line,
              baseline: offsetGeometry(line.baseline, deltaY),
            }
          : line,
      );
      setLines((segments) => applyLayoutLineGeometryToSegments(segments, nextLayoutLines));
      return { ...current, lines: nextLayoutLines };
    });
  }

  async function saveSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const selectedLine = layout.lines.find((line) => line.id === selectedLineId);
    if (!selectedLine) return;

    try {
      await api.updateLineGeometry(projectId, documentId, partId, selectedLineId, {
        baseline: selectedLine.baseline,
        mask: selectedLine.mask,
      });
      setLayout((current) => ({
        ...current,
        lines: current.lines.map((line) =>
          line.id === selectedLineId ? { ...line, manual_geometry: true } : line,
        ),
      }));
      setLines((current) =>
        current.map((line) =>
          line.id === selectedLineId
            ? {
                ...line,
                baseline: selectedLine.baseline ?? line.baseline,
                mask: selectedLine.mask ?? line.mask,
                manual_geometry: true,
              }
            : line,
        ),
      );
      setMutationError(null);
      setSaveMessage('Manual geometry saved');
      setSelectedLineSnapshot({
        baseline: selectedLine.baseline,
        mask: selectedLine.mask,
      });
    } catch (err) {
      if (selectedLineSnapshot) {
        setLayout((current) => ({
          ...current,
          lines: current.lines.map((line) =>
            line.id === selectedLineId
              ? {
                  ...line,
                  baseline: selectedLineSnapshot.baseline,
                  mask: selectedLineSnapshot.mask,
                }
              : line,
          ),
        }));
        setLines((current) =>
          applyLayoutLineGeometryToSegments(current, [
            {
              id: selectedLineId,
              baseline: selectedLineSnapshot.baseline,
              mask: selectedLineSnapshot.mask,
            },
          ]),
        );
      }
      setSaveMessage(null);
      setMutationError(layoutMutationMessage(err));
    }
  }

  async function resetSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const resetLayout = await api.resetPartLayout(projectId, documentId, partId, {
      line_ids: [selectedLineId],
    });
    const nextLayout = resetLayout ?? { blocks: [], lines: [] };
    setLayout(nextLayout);
    setLines((current) => applyLayoutLineGeometryToSegments(current, nextLayout.lines));
    setSelectedLineSnapshot(null);
    setSaveMessage('Layout reset');
  }

  async function replaceWithManualLine(kind: 'rectangle' | 'polygon', points: LinePoint[]) {
    if (!projectId || !documentId || !partId) return;
    const existing = sortedLines.map<LineUpsertRequest>(upsertLineRequest);
    const newLine: LineUpsertRequest = {
      order: existing.length,
      kind,
      points,
      source: 'manual',
    };
    const body: LinesReplaceRequest = {
      lines: [...existing, newLine],
    };
    try {
      const saved = await api.replacePartLines(projectId, documentId, partId, body);
      setLines(saved);
      setLayout((current) => syncLayoutLinesFromSegments(current, saved));
      setLineError(null);
      onDrawComplete();
    } catch (err) {
      setLineError(err instanceof Error ? err.message : 'Failed to save Segment geometry.');
    }
  }

  async function updateSegmentPoints(segmentId: string, points: LinePoint[]) {
    if (!projectId || !documentId || !partId) return;
    const cleanedPoints = cleanPolygonPoints(points);
    if (cleanedPoints.length < 3) {
      setLineError('A segment needs at least 3 points.');
      return;
    }
    const previousLines = lines;
    const optimisticLines = lines.map((line) =>
      line.id === segmentId ? { ...line, points: cleanedPoints, source: 'manual' as const } : line,
    );
    setLines(optimisticLines);
    setLayout((current) => syncLayoutLinesFromSegments(current, optimisticLines));
    try {
      const updatedLines = [...optimisticLines]
        .sort((a, b) => a.order - b.order)
        .map<LineUpsertRequest>((line, order) => upsertLineRequest(line, order));
      const saved = await api.replacePartLines(projectId, documentId, partId, {
        lines: updatedLines,
      });
      setLines(saved);
      setLayout((current) => syncLayoutLinesFromSegments(current, saved));
      setLineError(null);
    } catch (err) {
      setLines(previousLines);
      setLayout((current) => syncLayoutLinesFromSegments(current, previousLines));
      setLineError(layoutMutationMessage(err));
    }
  }

  async function moveSelectedSegmentRight() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const updatedLines = [...lines]
      .sort((a, b) => a.order - b.order)
      .map<LineUpsertRequest>((line, order) => ({
        ...upsertLineRequest(line, order),
        points:
          line.id === selectedSegmentId
            ? line.points.map(([x, y]) => [x + 5, y])
            : line.points,
      }));
    const saved = await api.replacePartLines(projectId, documentId, partId, {
      lines: updatedLines,
    });
    setLines(
      saved ??
        lines.map((line) =>
          line.id === selectedSegmentId
            ? { ...line, points: line.points.map(([x, y]) => [x + 5, y]) }
            : line,
        ),
    );
  }

  async function deleteSelectedSegment() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const remainingLines = [...lines]
      .sort((a, b) => a.order - b.order)
      .filter((line) => line.id !== selectedSegmentId)
      .map<LineUpsertRequest>(upsertLineRequest);
    const saved = await api.replacePartLines(projectId, documentId, partId, {
      lines: remainingLines,
    });
    setLines(saved ?? lines.filter((line) => line.id !== selectedSegmentId));
    setSelectedSegmentId(null);
  }

  async function runAutoSegment() {
    if (!projectId || !documentId || !partId) return;
    if (
      lines.length > 0 &&
      !window.confirm(
        'Run Kraken line segmentation? Existing machine Segments on this Page will be replaced.',
      )
    ) {
      return;
    }
    setSegmenting(true);
    setSegmentMessage(null);
    setPairingError(null);
    try {
      const enqueued = await api.segmentPart(projectId, documentId, partId, {
        use_otsu_refinement: useOtsuRefinement,
        otsu_sphere_radius: otsuSphereRadius,
      });
      if (trackJobAndWait) {
        await trackJobAndWait(enqueued.job_id, {
          label: 'Kraken line segmentation',
          kind: 'segmentation',
        });
      } else {
        await waitForJob(enqueued.job_id);
      }
      const [reloadedLines, reloadedLayout, pairing] = await Promise.all([
        api.listPartLines(projectId, documentId, partId),
        api.getPartLayout(projectId, documentId, partId),
        api.getPagePairing(projectId, documentId, partId),
      ]);
      setLines(reloadedLines);
      setLayout(reloadedLayout ?? { blocks: [], lines: [] });
      setSelectedLineId(null);
      setSelectedSegmentId(null);
      setSelectedLineSnapshot(null);
      setApprovedTextDraft('');
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setSegmentMessage(
        useOtsuRefinement
          ? `Kraken segmentation completed with Otsu refinement (${otsuSphereRadius}px sphere, ${reloadedLines.length} Segment(s)).`
          : `Kraken segmentation completed using raw Kraken boundaries (${reloadedLines.length} Segment(s)).`,
      );
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Auto segment failed.');
    } finally {
      setSegmenting(false);
    }
  }

  return {
    selectedLineId,
    setSelectedLineId,
    selectedLineSnapshot,
    setSelectedLineSnapshot,
    saveMessage,
    setSaveMessage,
    mutationError,
    segmenting,
    useOtsuRefinement,
    setUseOtsuRefinement,
    otsuSphereRadius,
    setOtsuSphereRadius,
    segmentMessage,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
    replaceWithManualLine,
    updateSegmentPoints,
    moveSelectedSegmentRight,
    deleteSelectedSegment,
    runAutoSegment,
  };
}
