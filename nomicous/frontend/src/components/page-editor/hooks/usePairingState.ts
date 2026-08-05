import {
  useEffect,
  useState,
  type ChangeEvent,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  api,
  type InferenceModelResponse,
  type JobResponse,
  type LineResponse,
  type TranscribeJobResult,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { fetchPartImage } from "../../../api/imageCache";
import {
  blobToBase64,
  registrySelectionFromArtifactRef,
  runLocalInference,
  type LocalInferenceCallbacks,
  type LocalRun,
  type TranscribeBatchRunOutput,
  isAbortError,
  isRunSupersededError,
  localOnlyRunFailedMessage,
  localOnlyUnavailableMessage,
  RunSupersededError,
} from "../../../inference";
import type { PageEditorJobKind } from "../jobProgress";
import {
  lineTextForLayer,
  modelLayerIdForPromotion,
  withLocalGroundTruth,
} from "./utils";

type PairingStateInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  lines: LineResponse[];
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  transcriptionLayers: TranscriptionLayerResponse[];
  setTranscriptionLayers: Dispatch<
    SetStateAction<TranscriptionLayerResponse[]>
  >;
  selectedTranscriptionLayerId: string | null;
  setSelectedTranscriptionLayerId: Dispatch<SetStateAction<string | null>>;
  groundTruthTranscriptionId: string | null;
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
  selectedTranscribeModelId: string | null;
  transcribeModels: InferenceModelResponse[];
  partImageUrl: string | null;
  shouldUseLocalPath: (registryModelId: string) => boolean;
  /** False under "Local only" routing: no cloud job may ever be enqueued. */
  cloudInferenceEnabled: boolean;
  localInference: LocalInferenceCallbacks;
  trackJobAndWait: (
    jobId: string,
    meta: { label: string; kind: PageEditorJobKind },
  ) => Promise<JobResponse>;
  trackLocalTask: <T>(
    meta: { label: string; kind: PageEditorJobKind },
    run: (signal: AbortSignal) => Promise<T>,
  ) => Promise<T>;
};

export function usePairingState({
  projectId,
  documentId,
  partId,
  lines,
  setLines,
  transcriptionLayers,
  setTranscriptionLayers,
  selectedTranscriptionLayerId,
  setSelectedTranscriptionLayerId,
  groundTruthTranscriptionId,
  setTextLines,
  setPairingProgress,
  setPairingError,
  selectedTranscribeModelId,
  transcribeModels,
  partImageUrl,
  shouldUseLocalPath,
  cloudInferenceEnabled,
  localInference,
  trackJobAndWait,
  trackLocalTask,
}: PairingStateInput) {
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(
    null,
  );
  const [approvedTextDraft, setApprovedTextDraft] = useState("");
  const [transcriptionSaveMessage, setTranscriptionSaveMessage] = useState<
    string | null
  >(null);
  const [ocrRunning, setOcrRunning] = useState(false);
  const [ocrScope, setOcrScope] = useState<"segment" | "page" | null>(null);
  const [ocrMessage, setOcrMessage] = useState<string | null>(null);

  useEffect(() => {
    setSelectedSegmentId(null);
    setApprovedTextDraft("");
    setTranscriptionSaveMessage(null);
    setOcrMessage(null);
  }, [projectId, documentId, partId]);

  const selectedSegmentIndex =
    selectedSegmentId === null
      ? null
      : [...lines]
          .sort((a, b) => a.order - b.order)
          .findIndex((line) => line.id === selectedSegmentId);

  const selectedSegmentNumber =
    selectedSegmentIndex === null || selectedSegmentIndex < 0
      ? null
      : selectedSegmentIndex + 1;

  const selectedTranscriptionLayer =
    selectedTranscriptionLayerId === null
      ? null
      : (transcriptionLayers.find(
          (layer) => layer.id === selectedTranscriptionLayerId,
        ) ?? null);

  const selectedSegment =
    selectedSegmentId === null
      ? null
      : (lines.find((line) => line.id === selectedSegmentId) ?? null);

  async function pairTextLine(order: number) {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    try {
      const pairing = await api.pairTextLine(projectId, documentId, partId, {
        line_id: selectedSegmentId,
        text_line_order: order,
      });
      const candidate = pairing.text_lines.find(
        (textLine) => textLine.order === order,
      );
      if (candidate) {
        // Read-modify-write after an await: fold into the current segments, not
        // into the render-time snapshot, or a concurrent edit is reverted.
        setLines((current) =>
          withLocalGroundTruth(
            current,
            groundTruthTranscriptionId,
            selectedSegmentId,
            candidate.text,
          ),
        );
        setApprovedTextDraft(candidate.text);
      }
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(
        err instanceof Error ? err.message : "Failed to pair Text line.",
      );
    }
  }

  async function saveApprovedText() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (!groundTruthTranscriptionId) {
      setPairingError("Ground truth transcription layer is not available.");
      return;
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        selectedSegmentId,
        { text: approvedTextDraft },
      );
      // Read-modify-write after an await: fold into the current segments, not
      // into the render-time snapshot, or a concurrent edit is reverted.
      setLines((current) =>
        withLocalGroundTruth(
          current,
          groundTruthTranscriptionId,
          selectedSegmentId,
          updated.text,
        ),
      );
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(
        err instanceof Error ? err.message : "Failed to save approved text.",
      );
    }
  }

  function selectTranscriptionLayer(event: ChangeEvent<HTMLSelectElement>) {
    const nextLayerId = event.target.value;
    setSelectedTranscriptionLayerId(nextLayerId);
    setTranscriptionSaveMessage(null);
    setPairingError(null);
    if (selectedSegment) {
      setApprovedTextDraft(lineTextForLayer(selectedSegment, nextLayerId));
    }
  }

  async function saveGroundTruthText() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (
      !groundTruthTranscriptionId ||
      selectedTranscriptionLayer?.kind !== "ground_truth"
    ) {
      setPairingError("Only Ground truth can be edited.");
      return;
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        selectedSegmentId,
        { text: approvedTextDraft },
      );
      // Read-modify-write after an await: fold into the current segments, not
      // into the render-time snapshot, or a concurrent edit is reverted.
      setLines((current) =>
        withLocalGroundTruth(
          current,
          groundTruthTranscriptionId,
          selectedSegmentId,
          updated.text,
        ),
      );
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
      setTranscriptionSaveMessage("Ground truth text saved");
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(
        err instanceof Error
          ? err.message
          : "Failed to save Ground truth text.",
      );
    }
  }

  /** Keeps the draft box on the open segment showing the layer just written. */
  function syncApprovedTextDraft(
    reloadedLines: LineResponse[],
    layerId: string,
  ) {
    if (!selectedSegmentId) return;
    const segment = reloadedLines.find((line) => line.id === selectedSegmentId);
    if (segment) {
      setApprovedTextDraft(lineTextForLayer(segment, layerId));
    }
  }

  async function refreshAfterOcr(modelLayerId: string) {
    if (!projectId || !documentId || !partId) return;
    const [reloadedLines, layers] = await Promise.all([
      api.listPartLines(projectId, documentId, partId),
      api.listTranscriptions(projectId, documentId),
    ]);
    setLines(reloadedLines);
    setTranscriptionLayers(layers);
    setSelectedTranscriptionLayerId(modelLayerId);
    syncApprovedTextDraft(reloadedLines, modelLayerId);
  }

  /**
   * The page state a finished transcription leaves behind, wherever it was
   * produced: the new layer is folded into the segments already on screen, then
   * the page reloads to pick up what the server actually stored.
   */
  async function applyTranscribeResult(result: TranscribeJobResult) {
    setLines((current) =>
      current.map((line) => {
        const output = result.lines.find((entry) => entry.line_id === line.id);
        if (!output) return line;
        const withoutLayer = line.line_transcriptions.filter(
          (transcription) =>
            transcription.transcription_id !== result.transcription_id,
        );
        return {
          ...line,
          line_transcriptions: [
            ...withoutLayer,
            {
              id: `ocr-${line.id}-${result.transcription_id}`,
              transcription_id: result.transcription_id,
              transcription_kind: "model" as const,
              text: output.text,
              confidence: output.confidence,
            },
          ],
        };
      }),
    );
    await refreshAfterOcr(result.transcription_id);
    return result;
  }

  async function applyTranscribeJob(job: JobResponse) {
    const result = job.result as TranscribeJobResult | null;
    if (!result?.transcription_id) {
      throw new Error("Transcribe job returned no result.");
    }
    return applyTranscribeResult(result);
  }

  function selectedTranscribeModel(): InferenceModelResponse | null {
    if (!selectedTranscribeModelId) return null;
    return (
      transcribeModels.find(
        (model) => model.id === selectedTranscribeModelId,
      ) ?? null
    );
  }

  async function loadPartImageBase64(): Promise<string> {
    if (!partImageUrl) {
      throw new Error("Page image is not available for local inference.");
    }
    const blob = await fetchPartImage(partImageUrl);
    return blobToBase64(blob);
  }

  /**
   * Runs the helper and persists what it produced, and nothing more.
   *
   * Everything in here is part of "did the local run produce a saved result": a
   * failure means the cloud still has to do the work. The refresh that follows a
   * success is deliberately left to the caller - see
   * `runLocalTranscribeWithFallback`.
   */
  async function runLocalTranscribe(
    lineIds: string[],
    signal: AbortSignal,
    onRunStarted: (run: LocalRun) => void,
  ): Promise<TranscribeJobResult> {
    const model = selectedTranscribeModel();
    if (!model) {
      throw new Error("Select an HTR model before running OCR.");
    }
    const { registryModelId, registryTag } = registrySelectionFromArtifactRef(
      model.artifact_ref,
    );
    if (!shouldUseLocalPath(registryModelId)) {
      throw new Error("Selected model is not eligible for local inference.");
    }

    const targetLines = lines
      .filter((line) => lineIds.includes(line.id))
      .sort((a, b) => a.order - b.order);
    if (targetLines.length === 0) {
      throw new Error("No matching segments to transcribe.");
    }

    const imageBytes = await loadPartImageBase64();
    signal.throwIfAborted();
    const run = localInference.startRun(registryModelId, registryTag);
    onRunStarted(run);
    try {
      const combinedSignal = AbortSignal.any([
        signal,
        run.cloudSwitchSignal,
        run.supersededSignal,
      ]);
      combinedSignal.throwIfAborted();
      const response = await runLocalInference({
        task: "transcribe",
        registry_model_id: registryModelId,
        registry_tag: registryTag,
        image_bytes: imageBytes,
        signal: combinedSignal,
        params: {
          lines: targetLines.map((line, index) => ({
            line_id: line.id,
            line_index: index,
            points: line.points,
          })),
        },
      });
      combinedSignal.throwIfAborted();

      if (response.task !== "transcribe" || !("lines" in response.output)) {
        throw new Error("Local transcribe returned an unexpected response.");
      }
      const batch = response.output as TranscribeBatchRunOutput;
      // A batch is now allowed to be a partial success, so drop the lines that
      // failed rather than reading `.text` off a null output. Persisting the
      // survivors is the point of the isolation: one bad line used to discard
      // the whole page.
      const transcribed = batch.lines.flatMap((entry) =>
        entry.output
          ? [
              {
                line_id: entry.line_id ?? targetLines[entry.line_index]?.id ?? "",
                text: entry.output.text,
                confidence: entry.output.confidence,
                character_confidences: entry.output.character_confidences,
              },
            ]
          : [],
      );
      if (transcribed.length === 0) {
        throw new Error("Local transcribe returned no usable lines.");
      }
      return await api.persistLocalTranscribe(
        projectId!,
        documentId!,
        partId!,
        {
          registry_model_id: registryModelId,
          registry_tag: registryTag,
          lines: transcribed,
        },
      );
    } finally {
      run.end();
    }
  }

  async function runLocalTranscribeWithFallback(
    lineIds: string[],
    jobMeta: { label: string; kind: PageEditorJobKind },
  ) {
    // The signal the jobs panel owns. Only an abort on *this* signal is a user
    // cancellation; the run's own signals say why else it stopped.
    let userCancelSignal: AbortSignal | undefined;
    // Owned by the page, so it stays readable after the run it belongs to has
    // already unwound.
    let supersededSignal: AbortSignal | undefined;
    // Set once the server has stored the local result. Only the work up to this
    // point may send the page to the cloud; anything after it would be paying
    // twice for a transcription that already exists.
    let persisted: TranscribeJobResult | null = null;
    try {
      persisted = await trackLocalTask(jobMeta, (signal) => {
        userCancelSignal = signal;
        return runLocalTranscribe(lineIds, signal, (run) => {
          supersededSignal = run.supersededSignal;
        });
      });
    } catch (err) {
      // A cancellation the user asked for must stop here, never continue
      // silently in the cloud.
      if (userCancelSignal?.aborted) throw err;
      // A newer run for this page owns the outcome now. Drop this one outright -
      // no banner, and above all no cloud job to race with it.
      if (supersededSignal?.aborted) throw new RunSupersededError();
      if (!cloudInferenceEnabled) {
        throw new Error(localOnlyRunFailedMessage(err));
      }
      // Any other local failure (weights missing, helper crash, 503, …) falls
      // through to the cloud path below.
    }

    if (persisted) {
      // Only the reload is left. A failure here is a stale view of a
      // transcription that is already saved, so it surfaces as an error banner
      // and stops - it is not a reason to run the page again in the cloud.
      return applyTranscribeResult(persisted);
    }

    const enqueued = await api.enqueueTranscribePart(
      projectId!,
      documentId!,
      partId!,
      {
        model_id: selectedTranscribeModelId!,
        line_ids: lineIds.length === lines.length ? undefined : lineIds,
      },
    );
    const job = await trackJobAndWait(enqueued.job_id, jobMeta);
    return applyTranscribeJob(job);
  }

  async function runSegmentOcr() {
    if (!projectId || !documentId || !partId) {
      setPairingError("Page context is missing. Reload and try again.");
      return;
    }
    if (!selectedSegmentId) {
      setPairingError("Select a segment on the canvas first.");
      return;
    }
    if (!selectedTranscribeModelId) {
      setPairingError("Select an HTR model before running OCR.");
      return;
    }
    setOcrRunning(true);
    setOcrScope("segment");
    setOcrMessage(null);
    setPairingError(null);
    try {
      const model = selectedTranscribeModel();
      const registryModelId = model
        ? registrySelectionFromArtifactRef(model.artifact_ref).registryModelId
        : null;
      if (model && registryModelId && shouldUseLocalPath(registryModelId)) {
        const result = await runLocalTranscribeWithFallback(
          [selectedSegmentId],
          {
            label: selectedSegmentNumber
              ? `Segment ${selectedSegmentNumber}`
              : "Selected segment",
            kind: "transcription-segment",
          },
        );
        const hasAnyText = result.lines.some((line) => line.text?.trim());
        setOcrMessage(
          hasAnyText
            ? "OCR prediction completed for selected Segment (local)."
            : "OCR finished with no text for this segment.",
        );
        return;
      }

      if (!cloudInferenceEnabled) {
        throw new Error(localOnlyUnavailableMessage());
      }

      const enqueued = await api.enqueueTranscribePart(
        projectId,
        documentId,
        partId,
        {
          model_id: selectedTranscribeModelId,
          line_ids: [selectedSegmentId],
        },
      );
      const job = await trackJobAndWait(enqueued.job_id, {
        label: selectedSegmentNumber
          ? `Segment ${selectedSegmentNumber}`
          : "Selected segment",
        kind: "transcription-segment",
      });
      const result = await applyTranscribeJob(job);
      const hasAnyText = result.lines.some((line) => line.text?.trim());
      setOcrMessage(
        hasAnyText
          ? "OCR prediction completed for selected Segment."
          : "OCR finished with no text for this segment.",
      );
    } catch (err) {
      // The jobs panel already reports a user cancellation, and a superseded run
      // is replaced by its successor; neither deserves an error banner.
      if (isAbortError(err) || isRunSupersededError(err)) return;
      setPairingError(
        err instanceof Error ? err.message : "Segment OCR failed.",
      );
    } finally {
      setOcrRunning(false);
      setOcrScope(null);
    }
  }

  async function runPageOcr() {
    if (!projectId || !documentId || !partId) {
      setPairingError("Page context is missing. Reload and try again.");
      return;
    }
    if (!selectedTranscribeModelId) {
      setPairingError("Select an HTR model before running OCR.");
      return;
    }
    setOcrRunning(true);
    setOcrScope("page");
    setOcrMessage(null);
    setPairingError(null);
    try {
      const model = selectedTranscribeModel();
      const registryModelId = model
        ? registrySelectionFromArtifactRef(model.artifact_ref).registryModelId
        : null;
      if (model && registryModelId && shouldUseLocalPath(registryModelId)) {
        const result = await runLocalTranscribeWithFallback(
          lines.map((line) => line.id),
          {
            label: "Full page",
            kind: "transcription-page",
          },
        );
        const withText = result.lines.filter((line) =>
          line.text?.trim(),
        ).length;
        setOcrMessage(
          withText > 0
            ? `OCR prediction completed for ${withText} Segment(s) (local).`
            : "OCR finished with no text for the selected segments.",
        );
        return;
      }

      if (!cloudInferenceEnabled) {
        throw new Error(localOnlyUnavailableMessage());
      }

      const enqueued = await api.enqueueTranscribePart(
        projectId,
        documentId,
        partId,
        {
          model_id: selectedTranscribeModelId,
        },
      );
      const job = await trackJobAndWait(enqueued.job_id, {
        label: "Full page",
        kind: "transcription-page",
      });
      const result = await applyTranscribeJob(job);
      const withText = result.lines.filter((line) => line.text?.trim()).length;
      setOcrMessage(
        withText > 0
          ? `OCR prediction completed for ${withText} Segment(s).`
          : "OCR finished with no text for the selected segments.",
      );
    } catch (err) {
      // The jobs panel already reports a user cancellation, and a superseded run
      // is replaced by its successor; neither deserves an error banner.
      if (isAbortError(err) || isRunSupersededError(err)) return;
      setPairingError(err instanceof Error ? err.message : "Page OCR failed.");
    } finally {
      setOcrRunning(false);
      setOcrScope(null);
    }
  }

  async function promoteSelectedSegmentToGroundTruth() {
    if (
      !projectId ||
      !documentId ||
      !partId ||
      !selectedSegmentId ||
      !selectedSegment
    )
      return;
    const modelLayerId = modelLayerIdForPromotion(
      selectedSegment,
      selectedTranscriptionLayer,
    );
    if (!modelLayerId) {
      setPairingError(
        "Model transcription is not available to save as Ground truth.",
      );
      return;
    }
    try {
      await api.copyToGroundTruth(projectId, documentId, modelLayerId, {
        line_ids: [selectedSegmentId],
      });
      const [reloadedLines, pairing] = await Promise.all([
        api.listPartLines(projectId, documentId, partId),
        api.getPagePairing(projectId, documentId, partId),
      ]);
      setLines(reloadedLines);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
      if (groundTruthTranscriptionId) {
        setSelectedTranscriptionLayerId(groundTruthTranscriptionId);
        syncApprovedTextDraft(reloadedLines, groundTruthTranscriptionId);
      }
      setTranscriptionSaveMessage("Saved to Ground truth");
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(
        err instanceof Error ? err.message : "Failed to save to Ground truth.",
      );
    }
  }

  function selectSegment(lineId: string) {
    const selected = lines.find((line) => line.id === lineId) ?? null;
    setSelectedSegmentId(lineId);
    setTranscriptionSaveMessage(null);
    setApprovedTextDraft(
      selected ? lineTextForLayer(selected, selectedTranscriptionLayerId) : "",
    );
  }

  function navigateSegment(direction: -1 | 1) {
    const sorted = [...lines].sort((a, b) => a.order - b.order);
    if (sorted.length === 0) return;

    const currentIndex = selectedSegmentId
      ? sorted.findIndex((line) => line.id === selectedSegmentId)
      : -1;

    let nextIndex: number;
    if (currentIndex < 0) {
      nextIndex = direction === 1 ? 0 : sorted.length - 1;
    } else {
      nextIndex = Math.min(
        Math.max(currentIndex + direction, 0),
        sorted.length - 1,
      );
    }

    if (nextIndex !== currentIndex) {
      selectSegment(sorted[nextIndex].id);
    }
  }

  return {
    selectedSegmentId,
    setSelectedSegmentId,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    ocrRunning,
    ocrScope,
    ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    pairTextLine,
    saveApprovedText,
    selectTranscriptionLayer,
    saveGroundTruthText,
    runSegmentOcr,
    runPageOcr,
    promoteSelectedSegmentToGroundTruth,
    selectSegment,
    navigateSegment,
  };
}
