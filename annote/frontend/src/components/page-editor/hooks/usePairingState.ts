import { useEffect, useState, type ChangeEvent, type Dispatch, type SetStateAction } from 'react';
import {
  api,
  type LineResponse,
  type TranscriptionLayerResponse,
} from '../../../api/client';
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
}: PairingStateInput) {
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [pageTranscriptionText, setPageTranscriptionText] = useState('');
  const [approvedTextDraft, setApprovedTextDraft] = useState('');
  const [transcriptionSaveMessage, setTranscriptionSaveMessage] = useState<string | null>(null);
  const [ocrRunning, setOcrRunning] = useState(false);
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

  async function runSegmentOcr() {
    if (!projectId || !documentId || !partId || !selectedSegmentId || !selectedTranscribeModelId) {
      return;
    }
    setOcrRunning(true);
    setOcrMessage(null);
    setPairingError(null);
    try {
      const result = await api.ocrPredictLine(
        projectId,
        documentId,
        partId,
        selectedSegmentId,
        { model_id: selectedTranscribeModelId },
      );
      await refreshAfterOcr(result.transcription_id);
      setOcrMessage('OCR prediction completed for selected Segment.');
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Segment OCR failed.');
    } finally {
      setOcrRunning(false);
    }
  }

  async function runPageOcr() {
    if (!projectId || !documentId || !partId || !selectedTranscribeModelId) return;
    setOcrRunning(true);
    setOcrMessage(null);
    setPairingError(null);
    try {
      const result = await api.ocrPredictPart(projectId, documentId, partId, {
        model_id: selectedTranscribeModelId,
      });
      await refreshAfterOcr(result.transcription_id);
      setOcrMessage(`OCR prediction completed for ${result.lines.length} Segment(s).`);
    } catch (err) {
      setPairingError(err instanceof Error ? err.message : 'Page OCR failed.');
    } finally {
      setOcrRunning(false);
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

  return {
    selectedSegmentId,
    setSelectedSegmentId,
    pageTranscriptionText,
    setPageTranscriptionText,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    ocrRunning,
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
  };
}
