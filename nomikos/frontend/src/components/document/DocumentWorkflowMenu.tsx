import { useEffect, useState } from "react";
import {
  api,
  type DocumentWorkflowCounts,
  type InferenceModelResponse,
} from "../../api/client";
import { ApiError } from "../../api/errors";
import {
  ActionMenu,
  ActionMenuCaption,
  ActionMenuConfirm,
  ActionMenuDivider,
  ActionMenuItem,
  ActionMenuSection,
  ActionMenuWarning,
} from "../ui/ActionMenu";
import { toast } from "../ui/toast";
import { PageEditorModelSelect } from "../page-editor/PageEditorModelSelect";
import { resolveSegmentModelId } from "../page-editor/segmentModelChoice";
import {
  TRANSCRIBE_MODEL_NAME,
  batchQueuedMessage,
  pageCountLabel,
} from "./documentActionCopy";

type DocumentWorkflowMenuProps = {
  projectId: string;
  documentId: string;
  /** Null while the counts are still loading, which disables every item. */
  counts: DocumentWorkflowCounts | null;
  disabled?: boolean;
  /** Jobs were queued, so the counts and the parts list are both behind. */
  onJobsQueued: () => void;
};

/**
 * Segment and transcribe, for the whole document at once.
 *
 * These are document-level acts: they change every page, so they belong on the
 * document, not on whichever page happens to be open in the editor. The page
 * editor keeps its own single-page versions of the same two jobs.
 */
export function DocumentWorkflowMenu({
  projectId,
  documentId,
  counts,
  disabled = false,
  onJobsQueued,
}: DocumentWorkflowMenuProps) {
  const [confirmingResegment, setConfirmingResegment] = useState(false);
  const [running, setRunning] = useState(false);
  const [segmentModels, setSegmentModels] = useState<InferenceModelResponse[]>(
    [],
  );
  /**
   * Always one of the catalog ids once the catalog loads; null only while
   * the list is empty or failed to load, when `model_id: null` is sent and
   * the backend resolves its own default.
   */
  const [selectedSegmentModelId, setSelectedSegmentModelId] = useState<
    string | null
  >(null);

  // The segment catalog for the picker, with the canonical row preselected.
  // A failed request leaves an empty list and does not block segmenting.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const catalog = await api.listInferenceModels();
        if (!cancelled) {
          const segment = catalog.filter((model) => model.task === "segment");
          setSegmentModels(segment);
          setSelectedSegmentModelId((current) =>
            resolveSegmentModelId(segment, current),
          );
        }
      } catch {
        if (!cancelled) setSegmentModels([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, documentId]);

  const total = counts?.total ?? 0;
  const unsegmented = counts?.unsegmented ?? 0;
  const unpaired = counts?.unpaired ?? 0;
  const busy = disabled || running || counts === null;
  const selectedSegmentModelName =
    segmentModels.find((model) => model.id === selectedSegmentModelId)?.name ??
    null;

  async function runSegment(scope: "unsegmented" | "all", close: () => void) {
    setRunning(true);
    try {
      const result = await api.enqueueDocumentSegment(projectId, documentId, {
        scope,
        model_id: selectedSegmentModelId,
      });
      toast.success(batchQueuedMessage(result));
      onJobsQueued();
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : "Could not queue segmentation",
      );
    } finally {
      setRunning(false);
      setConfirmingResegment(false);
    }
  }

  async function runTranscribe(close: () => void) {
    setRunning(true);
    try {
      const result = await api.enqueueDocumentTranscribe(
        projectId,
        documentId,
        { scope: "unpaired", model_id: null },
      );
      toast.success(batchQueuedMessage(result));
      onJobsQueued();
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : "Could not queue transcription",
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <ActionMenu
      label="Workflow"
      menuLabel="Document workflow"
      wide
      onOpenChange={(open) => {
        if (!open) setConfirmingResegment(false);
      }}
    >
      {(close) =>
        confirmingResegment ? (
          <ActionMenuConfirm
            destructive
            busy={running}
            question={`Re-segment every page (${pageCountLabel(total)})?`}
            detail={
              selectedSegmentModelName
                ? `Lines with approved text, a pairing or hand-drawn geometry stay. Every other line is redrawn and the model's unapproved text on it is discarded. There is no undo. Runs with ${selectedSegmentModelName}.`
                : "Lines with approved text, a pairing or hand-drawn geometry stay. Every other line is redrawn and the model's unapproved text on it is discarded. There is no undo."
            }
            confirmLabel={
              running ? "Queueing…" : `Yes, re-segment ${pageCountLabel(total)}`
            }
            onCancel={() => setConfirmingResegment(false)}
            onConfirm={() => void runSegment("all", close)}
          />
        ) : (
          <>
            <ActionMenuSection>Segment</ActionMenuSection>
            <PageEditorModelSelect
              label="Seg"
              ariaLabel="Segmentation model"
              models={segmentModels}
              selectedModelId={selectedSegmentModelId}
              onSelectedModelIdChange={setSelectedSegmentModelId}
              disabled={busy}
            />
            <ActionMenuItem
              label="Segment unsegmented pages"
              meta={String(unsegmented)}
              disabled={busy || unsegmented === 0}
              onSelect={() => void runSegment("unsegmented", close)}
            />
            <ActionMenuItem
              destructive
              label="Re-segment every page"
              meta={String(total)}
              disabled={busy || total === 0}
              onSelect={() => setConfirmingResegment(true)}
            />
            <ActionMenuWarning>
              Re-segmenting discards unapproved machine text on lines nobody has
              touched. Only the top item is safe.
            </ActionMenuWarning>
            <ActionMenuDivider />
            <ActionMenuSection>Transcribe</ActionMenuSection>
            <ActionMenuCaption>
              Model <strong>{TRANSCRIBE_MODEL_NAME}</strong>
            </ActionMenuCaption>
            <ActionMenuItem
              label="Transcribe unpaired pages"
              meta={String(unpaired)}
              disabled={busy || unpaired === 0}
              onSelect={() => void runTranscribe(close)}
            />
          </>
        )
      }
    </ActionMenu>
  );
}
