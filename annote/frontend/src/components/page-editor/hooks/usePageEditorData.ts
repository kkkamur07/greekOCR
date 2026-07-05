import { useEffect, useState } from 'react';
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type InferenceModelResponse,
  type LineResponse,
  type PartLayoutResponse,
  type TranscriptionLayerResponse,
} from '../../../api/client';
import { ApiError } from '../../../api/errors';

function accessMessage(error: ApiError): string {
  if (error.status === 403 || error.status === 404) {
    return 'This page is not available to your account.';
  }
  return error.message;
}

export function usePageEditorData(
  projectId: string | undefined,
  documentId: string | undefined,
  partId: string | undefined,
  onRouteChange?: () => void,
) {
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(null);
  const [part, setPart] = useState<DocumentPartResponse | null>(null);
  const [layout, setLayout] = useState<PartLayoutResponse>({ blocks: [], lines: [] });
  const [lines, setLines] = useState<LineResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const [lineError, setLineError] = useState<string | null>(null);
  const [transcriptionLayers, setTranscriptionLayers] = useState<TranscriptionLayerResponse[]>([]);
  const [selectedTranscriptionLayerId, setSelectedTranscriptionLayerId] = useState<string | null>(
    null,
  );
  const [groundTruthTranscriptionId, setGroundTruthTranscriptionId] = useState<string | null>(
    null,
  );
  const [textLines, setTextLines] = useState<
    { order: number; text: string; paired_line_id: string | null }[]
  >([]);
  const [pairingProgress, setPairingProgress] = useState({
    paired_lines: 0,
    total_lines: 0,
    percent: 0,
  });
  const [pairingError, setPairingError] = useState<string | null>(null);
  const [transcribeModels, setTranscribeModels] = useState<InferenceModelResponse[]>([]);
  const [selectedTranscribeModelId, setSelectedTranscribeModelId] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId || !documentId || !partId) {
      setLoading(false);
      setError('Page route is incomplete.');
      return;
    }

    setLoading(true);
    setError(null);
    setLayoutError(null);
    setLineError(null);
    setDocument(null);
    setPart(null);
    setLayout({ blocks: [], lines: [] });
    setLines([]);
    setTranscriptionLayers([]);
    setSelectedTranscriptionLayerId(null);
    setGroundTruthTranscriptionId(null);
    setTextLines([]);
    setPairingProgress({ paired_lines: 0, total_lines: 0, percent: 0 });
    setPairingError(null);
    onRouteChange?.();

    (async () => {
      try {
        const doc = await api.getDocument(projectId, documentId);
        const sortedParts = [...doc.parts].sort((a, b) => a.order - b.order);
        const selectedPart = sortedParts.find((item) => item.id === partId);
        if (!selectedPart) {
          setError('This document part was not found.');
          return;
        }
        setDocument(doc);
        setPart(selectedPart);
        try {
          const loadedLayout = await api.getPartLayout(projectId, documentId, partId);
          setLayout(loadedLayout ?? { blocks: [], lines: [] });
        } catch (err) {
          setLayoutError(
            err instanceof ApiError && (err.status === 403 || err.status === 404)
              ? 'Layout editing is not available for this page.'
              : err instanceof Error
                ? err.message
                : 'Failed to load layout.',
          );
        }
        try {
          setLines(await api.listPartLines(projectId, documentId, partId));
        } catch (err) {
          setLineError(
            err instanceof ApiError && (err.status === 403 || err.status === 404)
              ? 'Segment geometry is not available for this page.'
              : err instanceof Error
                ? err.message
                : 'Failed to load Segment geometry.',
          );
        }
        try {
          const layers = await api.listTranscriptions(projectId, documentId);
          const groundTruth = layers.find((layer) => layer.kind === 'ground_truth');
          setTranscriptionLayers(layers);
          setGroundTruthTranscriptionId(groundTruth?.id ?? null);
          setSelectedTranscriptionLayerId(groundTruth?.id ?? layers[0]?.id ?? null);
          const pairing = await api.getPagePairing(projectId, documentId, partId);
          setTextLines(pairing.text_lines);
          setPairingProgress(pairing.pairing_progress);
        } catch (err) {
          setPairingError(
            err instanceof ApiError && (err.status === 403 || err.status === 404)
              ? 'Pairing is not available for this page.'
              : err instanceof Error
                ? err.message
                : 'Failed to load Pairing progress.',
          );
        }
        try {
          let models: InferenceModelResponse[] = [];
          try {
            const catalog = await api.listInferenceModels();
            models = catalog.filter((model) => model.task === 'transcribe');
          } catch {
            models = [];
          }

          try {
            const resolved = await api.resolvePartModelBinding(
              projectId,
              documentId,
              partId,
              'transcribe',
            );
            setSelectedTranscribeModelId(resolved.model.id);
            if (!models.some((model) => model.id === resolved.model.id)) {
              models = [resolved.model, ...models];
            }
          } catch {
            setSelectedTranscribeModelId(models[0]?.id ?? null);
          }
          setTranscribeModels(models);
        } catch {
          setTranscribeModels([]);
          setSelectedTranscribeModelId(null);
        }
      } catch (err) {
        setError(err instanceof ApiError ? accessMessage(err) : 'Failed to load page.');
      } finally {
        setLoading(false);
      }
    })();
    // onRouteChange resets page-local UI state when route params change; omit from deps intentionally.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- route-keyed reset only
  }, [projectId, documentId, partId]);

  const partIndex =
    document && part
      ? [...document.parts].sort((a, b) => a.order - b.order).findIndex((item) => item.id === part.id) +
        1
      : null;

  return {
    document,
    setDocument,
    part,
    setPart,
    layout,
    setLayout,
    lines,
    setLines,
    loading,
    error,
    layoutError,
    lineError,
    setLineError,
    transcriptionLayers,
    setTranscriptionLayers,
    selectedTranscriptionLayerId,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    textLines,
    setTextLines,
    pairingProgress,
    setPairingProgress,
    pairingError,
    setPairingError,
    transcribeModels,
    selectedTranscribeModelId,
    setSelectedTranscribeModelId,
    partIndex,
  };
}
