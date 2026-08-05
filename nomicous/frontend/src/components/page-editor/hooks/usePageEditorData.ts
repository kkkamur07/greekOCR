import { useEffect, useState } from "react";
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type InferenceModelResponse,
  type LineResponse,
  type PartLayoutResponse,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { ApiError } from "../../../api/errors";
import { readResource } from "../../../api/resourceCache";
import { resourceTags } from "../../../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  redirectToLogin,
} from "../../../auth/session";

function accessMessage(error: ApiError): string {
  if (error.status === 401) {
    redirectToLogin();
    return "";
  }
  if (error.status === 403 || error.status === 404) {
    return "This page is not available to your account.";
  }
  return error.message;
}

/**
 * The banner for one part of the page that failed to load while the rest of it
 * succeeded. 403 and 404 both mean "not yours to see", which reads better as the
 * feature-specific sentence than as the raw API message.
 */
function partialLoadMessage(
  error: unknown,
  unavailable: string,
  fallback: string,
): string {
  if (
    error instanceof ApiError &&
    (error.status === 403 || error.status === 404)
  ) {
    return unavailable;
  }
  return error instanceof Error ? error.message : fallback;
}

function sortedParts(
  document: DocumentWithPartsResponse,
): DocumentPartResponse[] {
  return [...document.parts].sort((a, b) => a.order - b.order);
}

function resolvePart(
  document: DocumentWithPartsResponse,
  partId: string,
): DocumentPartResponse | null {
  return sortedParts(document).find((item) => item.id === partId) ?? null;
}

function canReuseDocument(
  document: DocumentWithPartsResponse | null | undefined,
  projectId: string,
  documentId: string,
): document is DocumentWithPartsResponse {
  return document?.project_id === projectId && document.id === documentId;
}

async function loadTranscribeModels(
  projectId: string,
  documentId: string,
  partId: string,
): Promise<{
  models: InferenceModelResponse[];
  selectedModelId: string | null;
}> {
  let models: InferenceModelResponse[] = [];
  try {
    const catalog = await api.listInferenceModels();
    models = catalog.filter((model) => model.task === "transcribe");
  } catch {
    models = [];
  }

  try {
    const resolved = await api.resolvePartModelBinding(
      projectId,
      documentId,
      partId,
      "transcribe",
    );
    if (!models.some((model) => model.id === resolved.model.id)) {
      models = [resolved.model, ...models];
    }
    return { models, selectedModelId: resolved.model.id };
  } catch {
    return { models, selectedModelId: models[0]?.id ?? null };
  }
}

export function usePageEditorData(
  projectId: string | undefined,
  documentId: string | undefined,
  partId: string | undefined,
  onRouteChange?: () => void,
  initialDocument?: DocumentWithPartsResponse | null,
) {
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(
    null,
  );
  const [part, setPart] = useState<DocumentPartResponse | null>(null);
  const [layout, setLayout] = useState<PartLayoutResponse>({
    blocks: [],
    lines: [],
  });
  const [lines, setLines] = useState<LineResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const [lineError, setLineError] = useState<string | null>(null);
  const [transcriptionLayers, setTranscriptionLayers] = useState<
    TranscriptionLayerResponse[]
  >([]);
  const [selectedTranscriptionLayerId, setSelectedTranscriptionLayerId] =
    useState<string | null>(null);
  const [groundTruthTranscriptionId, setGroundTruthTranscriptionId] = useState<
    string | null
  >(null);
  const [textLines, setTextLines] = useState<
    { order: number; text: string; paired_line_id: string | null }[]
  >([]);
  const [pairingProgress, setPairingProgress] = useState({
    paired_lines: 0,
    total_lines: 0,
    percent: 0,
  });
  const [pairingError, setPairingError] = useState<string | null>(null);
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  const [selectedTranscribeModelId, setSelectedTranscribeModelId] = useState<
    string | null
  >(null);

  useEffect(() => {
    if (!projectId || !documentId || !partId) {
      setLoading(false);
      setError("Page route is incomplete.");
      return;
    }
    if (!hasAccessToken()) {
      redirectToLogin();
      return;
    }

    let cancelled = false;
    const apply = <T>(setter: (value: T) => void, value: T) => {
      if (!cancelled) {
        setter(value);
      }
    };

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

    void (async () => {
      try {
        // Paging through a document re-runs this effect for every part, and the
        // document itself does not change between them; the cache is what stops
        // each page turn from refetching it.
        const doc = canReuseDocument(initialDocument, projectId, documentId)
          ? initialDocument
          : await readResource(
              ["document", projectId, documentId],
              [resourceTags.document(projectId, documentId)],
              () => api.getDocument(projectId, documentId),
            );
        if (cancelled) return;

        const selectedPart = resolvePart(doc, partId);
        if (!selectedPart) {
          apply(setError, "This document part was not found.");
          return;
        }

        apply(setDocument, doc);
        apply(setPart, selectedPart);

        const [
          layoutResult,
          linesResult,
          transcriptionsResult,
          pairingResult,
          modelsResult,
        ] = await Promise.allSettled([
          api.getPartLayout(projectId, documentId, partId),
          api.listPartLines(projectId, documentId, partId),
          api.listTranscriptions(projectId, documentId),
          api.getPagePairing(projectId, documentId, partId),
          loadTranscribeModels(projectId, documentId, partId),
        ]);
        if (cancelled) return;

        if (layoutResult.status === "fulfilled") {
          apply(setLayout, layoutResult.value ?? { blocks: [], lines: [] });
        } else {
          const err = layoutResult.reason;
          if (isUnauthorized(err)) {
            redirectToLogin();
            return;
          }
          apply(
            setLayoutError,
            partialLoadMessage(
              err,
              "Layout editing is not available for this page.",
              "Failed to load layout.",
            ),
          );
        }

        if (linesResult.status === "fulfilled") {
          apply(setLines, linesResult.value);
        } else {
          const err = linesResult.reason;
          if (isUnauthorized(err)) {
            redirectToLogin();
            return;
          }
          apply(
            setLineError,
            partialLoadMessage(
              err,
              "Segment geometry is not available for this page.",
              "Failed to load Segment geometry.",
            ),
          );
        }

        if (transcriptionsResult.status === "fulfilled") {
          const layers = transcriptionsResult.value;
          const groundTruth = layers.find(
            (layer) => layer.kind === "ground_truth",
          );
          apply(setTranscriptionLayers, layers);
          apply(setGroundTruthTranscriptionId, groundTruth?.id ?? null);
          apply(
            setSelectedTranscriptionLayerId,
            groundTruth?.id ?? layers[0]?.id ?? null,
          );
        } else {
          const err = transcriptionsResult.reason;
          if (isUnauthorized(err)) {
            redirectToLogin();
            return;
          }
          apply(
            setPairingError,
            partialLoadMessage(
              err,
              "Pairing is not available for this page.",
              "Failed to load Pairing progress.",
            ),
          );
        }

        if (pairingResult.status === "fulfilled") {
          apply(setTextLines, pairingResult.value.text_lines);
          apply(setPairingProgress, pairingResult.value.pairing_progress);
        } else {
          const err = pairingResult.reason;
          if (isUnauthorized(err)) {
            redirectToLogin();
            return;
          }
          apply(
            setPairingError,
            partialLoadMessage(
              err,
              "Pairing is not available for this page.",
              "Failed to load Pairing progress.",
            ),
          );
        }

        if (modelsResult.status === "fulfilled") {
          apply(setTranscribeModels, modelsResult.value.models);
          apply(
            setSelectedTranscribeModelId,
            modelsResult.value.selectedModelId,
          );
        } else {
          apply(setTranscribeModels, []);
          apply(setSelectedTranscribeModelId, null);
        }
      } catch (err) {
        if (isUnauthorized(err)) {
          redirectToLogin();
          return;
        }
        apply(
          setError,
          err instanceof ApiError ? accessMessage(err) : "Failed to load page.",
        );
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // onRouteChange resets page-local UI state when route params change; omit from deps intentionally.
    // initialDocument is only read on first mount for the current route key.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- route-keyed reset only
  }, [projectId, documentId, partId]);

  const partIndex =
    document && part
      ? sortedParts(document).findIndex((item) => item.id === part.id) + 1
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
