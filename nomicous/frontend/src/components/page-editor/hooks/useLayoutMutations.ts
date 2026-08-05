import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  api,
  type JobResponse,
  type LayoutLineResponse,
  type LinePoint,
  type LineResponse,
  type PartLayoutResponse,
} from "../../../api/client";
import { fetchPartImage } from "../../../api/imageCache";
import { ApiError } from "../../../api/errors";
import {
  blobToBase64,
  DEFAULT_SEGMENT_REGISTRY_MODEL_ID,
  runLocalInference,
  type LocalInferenceCallbacks,
  isAbortError,
  localOnlyRunFailedMessage,
  localOnlyUnavailableMessage,
} from "../../../inference";
import { cleanPolygonPoints, offsetGeometry } from "../canvasGeometry";
import {
  applyCanvasEdit,
  applyCanvasEditInverse,
  pushEditOntoStack,
  type CanvasEdit,
} from "../editUndo";
import { SEGMENT_JOB_TIMEOUT_MS, type PageEditorJobKind } from "../jobProgress";
import { nextSegmentOrder } from "../segmentNumbering";
import {
  applyLayoutLineGeometryToSegments,
  mergeSavedLine,
  syncLayoutLinesFromSegments,
} from "./utils";

function layoutMutationMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return "Only project members can edit layout.";
  }
  return error instanceof Error ? error.message : "Layout update failed.";
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
    SetStateAction<
      { order: number; text: string; paired_line_id: string | null }[]
    >
  >;
  setPairingProgress: Dispatch<
    SetStateAction<{
      paired_lines: number;
      total_lines: number;
      percent: number;
    }>
  >;
  setPairingError: Dispatch<SetStateAction<string | null>>;
  selectedSegmentId: string | null;
  setSelectedSegmentId: Dispatch<SetStateAction<string | null>>;
  setApprovedTextDraft: Dispatch<SetStateAction<string>>;
  onDrawComplete: () => void;
  partImageUrl: string | null;
  shouldUseLocalPath: (registryModelId: string) => boolean;
  /** False under "Local only" routing: no cloud job may ever be enqueued. */
  cloudInferenceEnabled: boolean;
  segmentRegistryModelId?: string | null;
  localInference: LocalInferenceCallbacks;
  trackJobAndWait: (
    jobId: string,
    meta: { label: string; kind: PageEditorJobKind },
    options?: { timeoutMs?: number },
  ) => Promise<JobResponse>;
  trackLocalTask: <T>(
    meta: { label: string; kind: PageEditorJobKind },
    run: (signal: AbortSignal) => Promise<T>,
  ) => Promise<T>;
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
  partImageUrl,
  shouldUseLocalPath,
  cloudInferenceEnabled,
  segmentRegistryModelId,
  localInference,
  trackJobAndWait,
  trackLocalTask,
}: LayoutMutationsInput) {
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [selectedLineSnapshot, setSelectedLineSnapshot] = useState<{
    baseline?: LayoutLineResponse["baseline"];
    mask?: LayoutLineResponse["mask"];
  } | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);
  // A count, not a flag: a superseded run unwinds while its successor is still
  // going, and must not report the page as idle on the way out.
  const [segmentRunCount, setSegmentRunCount] = useState(0);
  const segmenting = segmentRunCount > 0;
  const [useOtsuRefinement, setUseOtsuRefinement] = useState(false);
  const [otsuSphereRadius, setOtsuSphereRadius] = useState(4);
  const [segmentMessage, setSegmentMessage] = useState<string | null>(null);
  const undoStackRef = useRef<CanvasEdit[]>([]);
  const redoStackRef = useRef<CanvasEdit[]>([]);
  const [editUndoRevision, setEditUndoRevision] = useState(0);
  const linesRef = useRef(lines);
  linesRef.current = lines;

  useEffect(() => {
    setSelectedLineId(null);
    setSelectedLineSnapshot(null);
    setSaveMessage(null);
    setMutationError(null);
    setSegmentMessage(null);
    undoStackRef.current = [];
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, [projectId, documentId, partId]);

  const recordEdit = useCallback((edit: CanvasEdit) => {
    undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, []);

  const applyLocalLines = useCallback(
    (nextLines: LineResponse[]) => {
      setLines(nextLines);
      setLayout((current) => syncLayoutLinesFromSegments(current, nextLines));
    },
    [setLines, setLayout],
  );

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
      setLines((segments) =>
        applyLayoutLineGeometryToSegments(segments, nextLayoutLines),
      );
      return { ...current, lines: nextLayoutLines };
    });
  }

  async function saveSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const selectedLine = layout.lines.find(
      (line) => line.id === selectedLineId,
    );
    if (!selectedLine) return;

    try {
      await api.updateLineGeometry(
        projectId,
        documentId,
        partId,
        selectedLineId,
        {
          baseline: selectedLine.baseline,
          mask: selectedLine.mask,
        },
      );
      setLayout((current) => ({
        ...current,
        lines: current.lines.map((line) =>
          line.id === selectedLineId
            ? { ...line, manual_geometry: true }
            : line,
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
      setSaveMessage("Manual geometry saved");
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
    const resetLayout = await api.resetPartLayout(
      projectId,
      documentId,
      partId,
      {
        line_ids: [selectedLineId],
      },
    );
    const nextLayout = resetLayout ?? { blocks: [], lines: [] };
    setLayout(nextLayout);
    setLines((current) =>
      applyLayoutLineGeometryToSegments(current, nextLayout.lines),
    );
    setSelectedLineSnapshot(null);
    setSaveMessage("Layout reset");
  }

  async function replaceWithManualLine(
    kind: "rectangle" | "polygon",
    points: LinePoint[],
  ) {
    if (!projectId || !documentId || !partId) return;
    try {
      // Past every Segment on the Page as it stands now. The count would collide
      // with a survivor of a delete: the backend leaves the freed order behind.
      const saved = await api.createPartLine(projectId, documentId, partId, {
        order: nextSegmentOrder(linesRef.current),
        kind,
        points,
      });
      // Merge into the segments as they are *now*, not as they were when the
      // request went out: an edit that landed while it was in flight must
      // survive, the way undo/redo already reads `linesRef`.
      const nextLines = mergeSavedLine(linesRef.current, saved);
      applyLocalLines(nextLines);
      recordEdit({ kind: "create", line: saved });
      setLineError(null);
      onDrawComplete();
    } catch (err) {
      setLineError(
        err instanceof Error ? err.message : "Failed to save Segment geometry.",
      );
    }
  }

  async function updateSegmentPoints(segmentId: string, points: LinePoint[]) {
    if (!projectId || !documentId || !partId) return;
    const cleanedPoints = cleanPolygonPoints(points);
    if (cleanedPoints.length < 3) {
      setLineError("A segment needs at least 3 points.");
      return;
    }
    const previousLines = lines;
    const before =
      previousLines.find((line) => line.id === segmentId)?.points ?? null;
    if (!before) return;
    const pointsUnchanged =
      before.length === cleanedPoints.length &&
      before.every(
        (point, index) =>
          point[0] === cleanedPoints[index][0] &&
          point[1] === cleanedPoints[index][1],
      );
    if (pointsUnchanged) return;

    const optimisticLines = lines.map((line) =>
      line.id === segmentId
        ? { ...line, points: cleanedPoints, source: "manual" as const }
        : line,
    );
    applyLocalLines(optimisticLines);
    try {
      const saved = await api.patchPartLine(
        projectId,
        documentId,
        partId,
        segmentId,
        {
          points: cleanedPoints,
        },
      );
      // `optimisticLines` is the snapshot from before the request; merging into
      // it would revert any edit made while the patch was in flight.
      const nextLines = mergeSavedLine(linesRef.current, saved);
      applyLocalLines(nextLines);
      recordEdit({
        kind: "points",
        segmentId,
        before,
        after: cleanedPoints,
      });
      setLineError(null);
    } catch (err) {
      applyLocalLines(previousLines);
      setLineError(layoutMutationMessage(err));
    }
  }

  async function moveSelectedSegmentRight() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const selectedLine = lines.find((line) => line.id === selectedSegmentId);
    if (!selectedLine) return;
    const nextPoints = selectedLine.points.map(
      ([x, y]) => [x + 5, y] as LinePoint,
    );
    await updateSegmentPoints(selectedSegmentId, nextPoints);
  }

  async function deleteSelectedSegment() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (
      !window.confirm(
        "Delete this Segment? Its geometry and pairing on this Page will be removed.",
      )
    ) {
      return;
    }
    const deletedId = selectedSegmentId;
    const deletedLine = lines.find((line) => line.id === deletedId);
    if (!deletedLine) return;
    const previousLines = lines;
    const previousLayout = layout;
    const optimisticLines = lines.filter((line) => line.id !== deletedId);
    applyLocalLines(optimisticLines);
    setSelectedSegmentId(null);
    try {
      await api.deletePartLine(projectId, documentId, partId, deletedId);
      recordEdit({ kind: "delete", line: deletedLine });
      setLineError(null);
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
    } catch (err) {
      setLines(previousLines);
      setLayout(previousLayout);
      setLineError(layoutMutationMessage(err));
    }
  }

  async function undoEdit() {
    const edit = undoStackRef.current.pop();
    if (!edit || !projectId || !documentId || !partId) {
      if (edit) undoStackRef.current.push(edit);
      return;
    }
    const previous = linesRef.current;
    try {
      if (edit.kind === "points") {
        applyLocalLines(applyCanvasEditInverse(previous, edit));
        await api.patchPartLine(projectId, documentId, partId, edit.segmentId, {
          points: edit.before,
        });
        redoStackRef.current = pushEditOntoStack(redoStackRef.current, edit);
      } else if (edit.kind === "create") {
        applyLocalLines(applyCanvasEditInverse(previous, edit));
        await api.deletePartLine(projectId, documentId, partId, edit.line.id);
        if (selectedSegmentId === edit.line.id) setSelectedSegmentId(null);
        redoStackRef.current = pushEditOntoStack(redoStackRef.current, edit);
      } else {
        const saved = await api.createPartLine(projectId, documentId, partId, {
          order: edit.line.order,
          kind: edit.line.kind,
          points: edit.line.points,
        });
        const restored: CanvasEdit = { kind: "delete", line: saved };
        applyLocalLines(applyCanvasEditInverse(previous, restored));
        redoStackRef.current = pushEditOntoStack(
          redoStackRef.current,
          restored,
        );
      }
      setLineError(null);
      setEditUndoRevision((value) => value + 1);
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
    } catch (err) {
      undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
      applyLocalLines(previous);
      setLineError(layoutMutationMessage(err));
      setEditUndoRevision((value) => value + 1);
    }
  }

  async function redoEdit() {
    const edit = redoStackRef.current.pop();
    if (!edit || !projectId || !documentId || !partId) {
      if (edit) redoStackRef.current.push(edit);
      return;
    }
    const previous = linesRef.current;
    try {
      if (edit.kind === "points") {
        applyLocalLines(applyCanvasEdit(previous, edit));
        await api.patchPartLine(projectId, documentId, partId, edit.segmentId, {
          points: edit.after,
        });
        undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
      } else if (edit.kind === "create") {
        const saved = await api.createPartLine(projectId, documentId, partId, {
          order: edit.line.order,
          kind: edit.line.kind,
          points: edit.line.points,
        });
        const created: CanvasEdit = { kind: "create", line: saved };
        applyLocalLines(applyCanvasEdit(previous, created));
        undoStackRef.current = pushEditOntoStack(undoStackRef.current, created);
      } else {
        applyLocalLines(applyCanvasEdit(previous, edit));
        await api.deletePartLine(projectId, documentId, partId, edit.line.id);
        if (selectedSegmentId === edit.line.id) setSelectedSegmentId(null);
        undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
      }
      setLineError(null);
      setEditUndoRevision((value) => value + 1);
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
    } catch (err) {
      redoStackRef.current = pushEditOntoStack(redoStackRef.current, edit);
      applyLocalLines(previous);
      setLineError(layoutMutationMessage(err));
      setEditUndoRevision((value) => value + 1);
    }
  }

  /**
   * The page state that follows a finished segmentation, local or cloud: both
   * paths replace the same Segments, layout and pairing, and only the sentence
   * they report differs. Returns the Segment count for that sentence.
   *
   * Callers reach this only after the `projectId`/`documentId`/`partId` guard in
   * `runAutoSegment`.
   */
  async function reloadAfterSegmentation(): Promise<number> {
    const [reloadedLines, reloadedLayout, pairing] = await Promise.all([
      api.listPartLines(projectId!, documentId!, partId!),
      api.getPartLayout(projectId!, documentId!, partId!),
      api.getPagePairing(projectId!, documentId!, partId!),
    ]);
    setLines(reloadedLines);
    setLayout(reloadedLayout ?? { blocks: [], lines: [] });
    setSelectedLineId(null);
    setSelectedSegmentId(null);
    setSelectedLineSnapshot(null);
    setApprovedTextDraft("");
    setTextLines(pairing.text_lines);
    setPairingProgress(pairing.pairing_progress);
    return reloadedLines.length;
  }

  async function runAutoSegment() {
    if (!projectId || !documentId || !partId) return;
    if (
      lines.length > 0 &&
      !window.confirm(
        "Run Kraken line segmentation? Existing machine Segments on this Page will be replaced.",
      )
    ) {
      return;
    }
    setSegmentRunCount((count) => count + 1);
    setSegmentMessage(null);
    setPairingError(null);
    // The signal the jobs panel owns. Only an abort on *this* signal is a user
    // cancellation; the run's own signals say why else it stopped.
    let userCancelSignal: AbortSignal | undefined;
    // Owned by the page, so it stays readable after the run it belongs to has
    // already unwound.
    let supersededSignal: AbortSignal | undefined;
    try {
      const resolvedSegmentModelId =
        segmentRegistryModelId ?? DEFAULT_SEGMENT_REGISTRY_MODEL_ID;
      if (shouldUseLocalPath(resolvedSegmentModelId)) {
        // Set once the server has stored the local segmentation. Only the work
        // up to that point may send the page to the cloud; running it again
        // afterwards would replace Segments that are already saved.
        let persistedLocally = false;
        try {
          await trackLocalTask(
            {
              label: "Kraken line segmentation",
              kind: "segmentation",
            },
            async (signal) => {
              userCancelSignal = signal;
              if (!partImageUrl) {
                throw new Error(
                  "Page image is not available for local segmentation.",
                );
              }
              const run = localInference.startRun(resolvedSegmentModelId);
              supersededSignal = run.supersededSignal;
              try {
                const combinedSignal = AbortSignal.any([
                  signal,
                  run.cloudSwitchSignal,
                  run.supersededSignal,
                ]);
                const imageBytes = await blobToBase64(
                  await fetchPartImage(partImageUrl),
                );
                combinedSignal.throwIfAborted();
                const response = await runLocalInference({
                  task: "segment",
                  registry_model_id: resolvedSegmentModelId,
                  image_bytes: imageBytes,
                  signal: combinedSignal,
                  params: {
                    use_otsu_refinement: useOtsuRefinement,
                    otsu_sphere_radius: otsuSphereRadius,
                  },
                });
                combinedSignal.throwIfAborted();
                if (response.task !== "segment") {
                  throw new Error(
                    "Local segment returned an unexpected response.",
                  );
                }
                await api.persistLocalSegment(projectId, documentId, partId, {
                  registry_model_id: resolvedSegmentModelId,
                  output: response.output,
                });
              } finally {
                run.end();
              }
            },
          );
          persistedLocally = true;
        } catch (err) {
          // A cancellation the user asked for must stop here, never continue
          // silently in the cloud.
          if (userCancelSignal?.aborted) throw err;
          // A newer run for this page owns the outcome now. Drop this one
          // outright - no banner, and above all no cloud job to race with it.
          if (supersededSignal?.aborted) return;
          if (!cloudInferenceEnabled) {
            throw new Error(localOnlyRunFailedMessage(err));
          }
          // Any other local failure (weights missing, helper crash, 503, …)
          // falls through to the cloud path below.
        }

        if (persistedLocally) {
          // Only the reload is left. A failure here is a stale view of a
          // segmentation that is already saved, so it surfaces as an error
          // banner and stops - it is not a reason to segment again in the cloud.
          const segmentCount = await reloadAfterSegmentation();
          setSegmentMessage(
            useOtsuRefinement
              ? `Kraken segmentation completed locally with Otsu refinement (${otsuSphereRadius}px sphere, ${segmentCount} Segment(s)).`
              : `Kraken segmentation completed locally using raw Kraken boundaries (${segmentCount} Segment(s)).`,
          );
          return;
        }
      }

      if (!cloudInferenceEnabled) {
        throw new Error(localOnlyUnavailableMessage());
      }

      const enqueued = await api.segmentPart(projectId, documentId, partId, {
        use_otsu_refinement: useOtsuRefinement,
        otsu_sphere_radius: otsuSphereRadius,
      });
      await trackJobAndWait(
        enqueued.job_id,
        {
          label: "Kraken line segmentation",
          kind: "segmentation",
        },
        { timeoutMs: SEGMENT_JOB_TIMEOUT_MS },
      );
      const segmentCount = await reloadAfterSegmentation();
      setSegmentMessage(
        useOtsuRefinement
          ? `Kraken segmentation completed with Otsu refinement (${otsuSphereRadius}px sphere, ${segmentCount} Segment(s)).`
          : `Kraken segmentation completed using raw Kraken boundaries (${segmentCount} Segment(s)).`,
      );
    } catch (err) {
      // The jobs panel already reports a user cancellation; no error banner.
      if (isAbortError(err) && userCancelSignal?.aborted) {
        return;
      }
      setPairingError(
        err instanceof Error ? err.message : "Auto segment failed.",
      );
    } finally {
      setSegmentRunCount((count) => Math.max(0, count - 1));
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
    undoEdit,
    redoEdit,
    canUndo: undoStackRef.current.length > 0,
    canRedo: redoStackRef.current.length > 0,
    editUndoRevision,
    runAutoSegment,
  };
}
