import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
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
import { queryClient, taggedMeta } from "../../../api/queryClient";
import { resourceTags } from "../../../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  redirectToLogin,
} from "../../../auth/session";
import { useBackgroundJobs } from "../../../context/BackgroundJobsContext";

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

type PartContentSetters = {
  setLayout: Dispatch<SetStateAction<PartLayoutResponse>>;
  setLayoutError: Dispatch<SetStateAction<string | null>>;
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  setLineError: Dispatch<SetStateAction<string | null>>;
  setTranscriptionLayers: Dispatch<
    SetStateAction<TranscriptionLayerResponse[]>
  >;
  setGroundTruthTranscriptionId: Dispatch<SetStateAction<string | null>>;
  setSelectedTranscriptionLayerId: Dispatch<SetStateAction<string | null>>;
  setPairingError: Dispatch<SetStateAction<string | null>>;
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
  setTranscribeModels: Dispatch<SetStateAction<InferenceModelResponse[]>>;
  setSelectedTranscribeModelId: Dispatch<SetStateAction<string | null>>;
};

/**
 * The layout/lines/transcriptions/pairing/models read for one part.
 *
 * Shared by the route-keyed mount effect below and the job-completion refresh
 * effect: the first runs it once resolving a fresh part, the second re-runs it
 * verbatim when a segmentation or OCR job finishes for the part already on
 * screen. `apply` is the caller's own cancelled/stale guard - this function
 * does not know or care which one it was given.
 */
async function fetchPartContent(
  projectId: string,
  documentId: string,
  partId: string,
  apply: <T>(setter: (value: T) => void, value: T) => void,
  setters: PartContentSetters,
): Promise<void> {
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

  if (layoutResult.status === "fulfilled") {
    apply(setters.setLayout, layoutResult.value ?? { blocks: [], lines: [] });
  } else {
    const err = layoutResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setLayoutError,
      partialLoadMessage(
        err,
        "Layout editing is not available for this page.",
        "Failed to load layout.",
      ),
    );
  }

  if (linesResult.status === "fulfilled") {
    apply(setters.setLines, linesResult.value);
  } else {
    const err = linesResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setLineError,
      partialLoadMessage(
        err,
        "Segment geometry is not available for this page.",
        "Failed to load Segment geometry.",
      ),
    );
  }

  if (transcriptionsResult.status === "fulfilled") {
    const layers = transcriptionsResult.value;
    const groundTruth = layers.find((layer) => layer.kind === "ground_truth");
    apply(setters.setTranscriptionLayers, layers);
    apply(setters.setGroundTruthTranscriptionId, groundTruth?.id ?? null);
    apply(
      setters.setSelectedTranscriptionLayerId,
      groundTruth?.id ?? layers[0]?.id ?? null,
    );
  } else {
    const err = transcriptionsResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setPairingError,
      partialLoadMessage(
        err,
        "Pairing is not available for this page.",
        "Failed to load Pairing progress.",
      ),
    );
  }

  if (pairingResult.status === "fulfilled") {
    apply(setters.setTextLines, pairingResult.value.text_lines);
    apply(setters.setPairingProgress, pairingResult.value.pairing_progress);
  } else {
    const err = pairingResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setPairingError,
      partialLoadMessage(
        err,
        "Pairing is not available for this page.",
        "Failed to load Pairing progress.",
      ),
    );
  }

  if (modelsResult.status === "fulfilled") {
    apply(setters.setTranscribeModels, modelsResult.value.models);
    apply(
      setters.setSelectedTranscribeModelId,
      modelsResult.value.selectedModelId,
    );
  } else {
    apply(setters.setTranscribeModels, []);
    apply(setters.setSelectedTranscribeModelId, null);
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

  /**
   * Which read of this part is the newest, counted across every effect that
   * writes the part's state rather than per effect.
   *
   * Two of them do: the route-mount load below and the job-completion refresh
   * after it, both filling the same setters from their own request. Guarding
   * them separately leaves the case where a job finishes while the first load
   * is still in flight - the refresh lands the new Segments, then the older
   * response overwrites them, and the page sits on pre-job state until a hard
   * reload. One counter, so the loser of that race stays quiet.
   */
  const contentGenerationRef = useRef(0);

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
    const generation = ++contentGenerationRef.current;
    const apply = <T>(setter: (value: T) => void, value: T) => {
      if (!cancelled && generation === contentGenerationRef.current) {
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
          : await queryClient.fetchQuery({
              queryKey: ["document", projectId, documentId],
              queryFn: () => api.getDocument(projectId, documentId),
              meta: taggedMeta([resourceTags.document(projectId, documentId)]),
            });
        if (cancelled) return;

        const selectedPart = resolvePart(doc, partId);
        if (!selectedPart) {
          apply(setError, "This document part was not found.");
          return;
        }

        apply(setDocument, doc);
        apply(setPart, selectedPart);
        if (cancelled) return;

        await fetchPartContent(projectId, documentId, partId, apply, {
          setLayout,
          setLayoutError,
          setLines,
          setLineError,
          setTranscriptionLayers,
          setGroundTruthTranscriptionId,
          setSelectedTranscriptionLayerId,
          setPairingError,
          setTextLines,
          setPairingProgress,
          setTranscribeModels,
          setSelectedTranscribeModelId,
        });
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

  const { subscribeToJobCompletion } = useBackgroundJobs();

  /**
   * The gap this closes: the mount effect above re-syncs the page from a
   * promise held by the one component instance whose button was clicked. If
   * that continuation never runs against a live instance - the tab was
   * backgrounded and its timers throttled, this component remounted, or the
   * researcher navigated away and back mid-job - nothing else re-syncs, and
   * only a hard reload recovers. A job going "done" is announced through
   * BackgroundJobsContext regardless of who started it or whether they are
   * still around to see it; this effect is what makes that announcement
   * useful to whichever instance is mounted and showing the affected part
   * right now.
   */
  useEffect(() => {
    if (!projectId || !documentId || !partId) return;

    let cancelled = false;

    const unsubscribe = subscribeToJobCompletion((event) => {
      if (cancelled) return;
      // Only "done" says there is anything new to read; the context is not
      // expected to announce failed or cancelled runs, but nothing here
      // should rely on that rather than say so itself.
      if (event.status !== "done") return;
      // Not this part's job - the instance actually showing that part (if any
      // is mounted) gets its own event.
      if (event.documentPartId !== partId) return;

      const generation = ++contentGenerationRef.current;
      const apply = <T>(setter: (value: T) => void, value: T) => {
        if (cancelled || generation !== contentGenerationRef.current) return;
        setter(value);
      };

      void fetchPartContent(projectId, documentId, partId, apply, {
        setLayout,
        setLayoutError,
        setLines,
        setLineError,
        setTranscriptionLayers,
        setGroundTruthTranscriptionId,
        setSelectedTranscriptionLayerId,
        setPairingError,
        setTextLines,
        setPairingProgress,
        setTranscribeModels,
        setSelectedTranscribeModelId,
      });
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [projectId, documentId, partId, subscribeToJobCompletion]);

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
