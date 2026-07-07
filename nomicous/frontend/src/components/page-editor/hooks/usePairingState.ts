import { useEffect, useState, type ChangeEvent, type Dispatch, type SetStateAction } from 'react';
import {
  api,
  type JobResponse,
  type LineResponse,
  type TranscribeJobResult,
  type TranscriptionLayerResponse,
} from '../../../api/client';
import type { PageEditorJobKind } from '../jobProgress';
import {
  lineTextForLayer,
  modelLayerIdForPromotion,
  withLocalGroundTruth,
} from './utils';

type PairingStateInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  lines: LineResponse[];
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  transcriptionLayers: TranscriptionLayerResponse[];
  setTranscriptionLayers: Dispatch<SetStateAction<TranscriptionLayerResponse[]>>;
  selectedTranscriptionLayerId: string | null;
  setSelectedTranscriptionLayerId: Dispatch<SetStateAction<string | null>>;
  groundTruthTranscriptionId: string | null;
  setTextLines: Dispatch<
    SetStateAction<{ order: number; text: string; paired_line_id: string | null }[]>
  >;
  setPairingProgress: Dispatch<
    SetStateAction<{ paired_lines: number; total_lines: number; percent: number }>
  >;
  setPairingError: Dispatch<SetStateAction<string | null>>;
  selectedTranscribeModelId: string | null;
  trackJobAndWait: (
    jobId: string,
    meta: { label: string; kind: PageEditorJobKind },
  ) => Promise<JobResponse>;
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
  trackJobAndWait,
}: PairingStateInput) {
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [pageTranscriptionText, setPageTranscriptionText] = useState('');
  const [approvedTextDraft, setApprovedTextDraft] = useState('');
  const [transcriptionSaveMessage, setTranscriptionSaveMessage] = useState<string | null>(null);
  const [ocrRunning, setOcrRunning] = useState(false);
  const [ocrScope, setOcrScope] = useState<'segment' | 'page' | null>(null);
  const [ocrMessage, setOcrMessage] = useState<string | null>(null);

  useEffect(() => {
    setSelectedSegmentId(null);
    setPageTranscriptionText('');
    setApprovedTextDraft('');
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
    selectedSegmentIndex === null || selectedSegmentIndex < 0 ? null : selectedSegmentIndex + 1;

  const selectedTranscriptionLayer =
    selectedTranscriptionLayerId === null
      ? null
      : (transcriptionLayers.find((layer) => layer.id === selectedTranscriptionLayerId) ?? null);

  const selectedSegment =
    selectedSegmentId === null ? null : (lines.find((line) => line.id === selectedSegmentId) ?? null);

  async function importPageTranscription() {
    if (!projectId || !documentId || !partId) return;
    try {
      const pairing = await api.importPageTranscription(projectId, documentId, partId, {
        text: pageTranscriptionText,
      });
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Failed to import Page transcription.');
    }
  }

  async function pairTextLine(order: number) {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    try {
      const pairing = await api.pairTextLine(projectId, documentId, partId, {
        line_id: selectedSegmentId,
        text_line_order: order,
      });
      const candidate = pairing.text_lines.find((textLine) => textLine.order === order);
      if (candidate) {
        setLines(withLocalGroundTruth(lines, groundTruthTranscriptionId, selectedSegmentId, candidate.text));
        setApprovedTextDraft(candidate.text);
      }
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Failed to pair Text line.');
    }
  }

  async function saveApprovedText() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (!groundTruthTranscriptionId) {
      setPairingError('Ground truth transcription layer is not available.');
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
      setLines(withLocalGroundTruth(lines, groundTruthTranscriptionId, selectedSegmentId, updated.text));
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Failed to save approved text.');
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
    if (!groundTruthTranscriptionId || selectedTranscriptionLayer?.kind !== 'ground_truth') {
      setPairingError('Only Ground truth can be edited.');
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
      setLines(withLocalGroundTruth(lines, groundTruthTranscriptionId, selectedSegmentId, updated.text));
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      setPairingError(null);
      setTranscriptionSaveMessage('Ground truth text saved');
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(err instanceof Error ? err.message : 'Failed to save Ground truth text.');
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
    if (selectedSegmentId) {
      const segment = reloadedLines.find((line) => line.id === selectedSegmentId);
      if (segment) {
        setApprovedTextDraft(lineTextForLayer(segment, modelLayerId));
      }
    }
  }

  async function applyTranscribeResult(job: JobResponse) {
    const result = job.result as TranscribeJobResult | null;
    if (!result?.transcription_id) {
      throw new Error('Transcribe job returned no result.');
    }
    setLines((current) =>
      current.map((line) => {
        const output = result.lines.find((entry) => entry.line_id === line.id);
        if (!output) return line;
        const withoutLayer = line.line_transcriptions.filter(
          (transcription) => transcription.transcription_id !== result.transcription_id,
        );
        return {
          ...line,
          line_transcriptions: [
            ...withoutLayer,
            {
              id: `ocr-${line.id}-${result.transcription_id}`,
              transcription_id: result.transcription_id,
              transcription_kind: 'model' as const,
              text: output.text,
              confidence: output.confidence,
              text_source: 'model' as const,
              character_confidences: null,
            },
          ],
        };
      }),
    );
    await refreshAfterOcr(result.transcription_id);
    return result;
  }

  async function runSegmentOcr() {
    if (!projectId || !documentId || !partId) {
      setPairingError('Page context is missing — reload and try again.');
      return;
    }
    if (!selectedSegmentId) {
      setPairingError('Select a segment on the canvas first.');
      return;
    }
    if (!selectedTranscribeModelId) {
      setPairingError('Select an HTR model before running OCR.');
      return;
    }
    setOcrRunning(true);
    setOcrScope('segment');
    setOcrMessage(null);
    setPairingError(null);
    try {
      const enqueued = await api.enqueueTranscribePart(projectId, documentId, partId, {
        model_id: selectedTranscribeModelId,
        line_ids: [selectedSegmentId],
      });
      const job = await trackJobAndWait(enqueued.job_id, {
        label: selectedSegmentNumber
          ? `Segment ${selectedSegmentNumber}`
          : 'Selected segment',
        kind: 'transcription-segment',
      });
      const result = await applyTranscribeResult(job);
      const hasAnyText = result.lines.some((line) => line.text?.trim());
      setOcrMessage(
        hasAnyText
          ? 'OCR prediction completed for selected Segment.'
          : 'OCR finished with no text for this segment.',
      );
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Segment OCR failed.');
    } finally {
      setOcrRunning(false);
      setOcrScope(null);
    }
  }

  async function runPageOcr() {
    if (!projectId || !documentId || !partId) {
      setPairingError('Page context is missing — reload and try again.');
      return;
    }
    if (!selectedTranscribeModelId) {
      setPairingError('Select an HTR model before running OCR.');
      return;
    }
    setOcrRunning(true);
    setOcrScope('page');
    setOcrMessage(null);
    setPairingError(null);
    try {
      const enqueued = await api.enqueueTranscribePart(projectId, documentId, partId, {
        model_id: selectedTranscribeModelId,
      });
      const job = await trackJobAndWait(enqueued.job_id, {
        label: 'Full page',
        kind: 'transcription-page',
      });
      const result = await applyTranscribeResult(job);
      const withText = result.lines.filter((line) => line.text?.trim()).length;
      setOcrMessage(
        withText > 0
          ? `OCR prediction completed for ${withText} Segment(s).`
          : 'OCR finished with no text for the selected segments.',
      );
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Page OCR failed.');
    } finally {
      setOcrRunning(false);
      setOcrScope(null);
    }
  }

  async function promoteSelectedSegmentToGroundTruth() {
    if (!projectId || !documentId || !partId || !selectedSegmentId || !selectedSegment) return;
    const modelLayerId = modelLayerIdForPromotion(selectedSegment, selectedTranscriptionLayer);
    if (!modelLayerId) {
      setPairingError('Model transcription is not available to save as Ground truth.');
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
        const refreshedSegment = reloadedLines.find((line) => line.id === selectedSegmentId);
        if (refreshedSegment) {
          setApprovedTextDraft(lineTextForLayer(refreshedSegment, groundTruthTranscriptionId));
        }
      }
      setTranscriptionSaveMessage('Saved to Ground truth');
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(err instanceof Error ? err.message : 'Failed to save to Ground truth.');
    }
  }

  function selectSegment(lineId: string) {
    const selected = lines.find((line) => line.id === lineId) ?? null;
    setSelectedSegmentId(lineId);
    setTranscriptionSaveMessage(null);
    setApprovedTextDraft(
      selected ? lineTextForLayer(selected, selectedTranscriptionLayerId) : '',
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
      nextIndex = Math.min(Math.max(currentIndex + direction, 0), sorted.length - 1);
    }

    if (nextIndex !== currentIndex) {
      selectSegment(sorted[nextIndex].id);
    }
  }

  return {
    selectedSegmentId,
    setSelectedSegmentId,
    pageTranscriptionText,
    setPageTranscriptionText,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    ocrRunning,
    ocrScope,
    ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    lineTextForLayer,
    importPageTranscription,
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
