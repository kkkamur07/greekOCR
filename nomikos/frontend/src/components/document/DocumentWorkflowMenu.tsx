import { useEffect, useState } from "react";
import {
  api,
  type DocumentWorkflowCounts,
  type InferenceModelResponse,
} from "../../api/client";
import { ApiError } from "../../api/errors";
import {
  ActionMenu,
  ActionMenuConfirm,
  ActionMenuDivider,
  ActionMenuItem,
  ActionMenuSection,
} from "../ui/ActionMenu";
import { toast } from "../ui/toast";
import { PageEditorModelSelect } from "../page-editor/PageEditorModelSelect";
import { useProjectModelDefaults } from "../page-editor/projectModelDefaults";
import { resolveSegmentModelId } from "../page-editor/segmentModelChoice";
import { resolveTranscribeModelId } from "../page-editor/transcribeModelChoice";
import { batchQueuedMessage, pageCountLabel } from "./documentActionCopy";

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
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  /**
   * The researcher's own pick, when they made one. Null means "no explicit
   * choice": the selection below follows the project binding, then the
   * canonical row (segment) or the first row, so a saved default moves every
   * picker that nobody overrode.
   */
  const [explicitSegmentModelId, setExplicitSegmentModelId] = useState<
    string | null
  >(null);
  const [explicitTranscribeModelId, setExplicitTranscribeModelId] = useState<
    string | null
  >(null);
  const projectDefaults = useProjectModelDefaults(projectId);

  /**
   * Always one of the catalog ids once the catalog loads; null only while
   * the list is empty or failed to load, when `model_id: null` is sent and
   * the backend resolves its own default.
   */
  const selectedSegmentModelId = resolveSegmentModelId(
    segmentModels,
    explicitSegmentModelId,
    projectDefaults.defaultModelId("segment"),
  );
  const selectedTranscribeModelId = resolveTranscribeModelId(
    transcribeModels,
    explicitTranscribeModelId,
    projectDefaults.defaultModelId("transcribe"),
  );

  // The catalog for both pickers. A failed request leaves empty lists and
  // does not block queueing: the requests then omit `model_id`.
  useEffect(() => {
    let cancelled = false;
    setExplicitSegmentModelId(null);
    setExplicitTranscribeModelId(null);
    void (async () => {
      try {
        const catalog = await api.listInferenceModels();
        if (!cancelled) {
          setSegmentModels(catalog.filter((model) => model.task === "segment"));
          setTranscribeModels(
            catalog.filter((model) => model.task === "transcribe"),
          );
        }
      } catch {
        if (!cancelled) {
          setSegmentModels([]);
          setTranscribeModels([]);
        }
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
        { scope: "unpaired", model_id: selectedTranscribeModelId },
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
            <div
              className="action-menu__model"
              role="group"
              aria-label="Segment"
            >
              <PageEditorModelSelect
                label="Model"
                ariaLabel="Segmentation model"
                models={segmentModels}
                selectedModelId={selectedSegmentModelId}
                onSelectedModelIdChange={setExplicitSegmentModelId}
                disabled={busy}
              />
            </div>
            <ActionMenuItem
              label="Segment unsegmented pages"
              meta={pageCountLabel(unsegmented)}
              quietMeta
              disabled={busy || unsegmented === 0}
              onSelect={() => void runSegment("unsegmented", close)}
            />
            <ActionMenuItem
              destructive
              label="Re-segment every page"
              detail="Discards unapproved machine text on untouched lines."
              meta={pageCountLabel(total)}
              quietMeta
              disabled={busy || total === 0}
              onSelect={() => setConfirmingResegment(true)}
            />
            <ActionMenuDivider />
            <ActionMenuSection>Transcribe</ActionMenuSection>
            <div
              className="action-menu__model"
              role="group"
              aria-label="Transcribe"
            >
              <PageEditorModelSelect
                label="Model"
                ariaLabel="HTR transcription model"
                models={transcribeModels}
                selectedModelId={selectedTranscribeModelId}
                onSelectedModelIdChange={setExplicitTranscribeModelId}
                disabled={busy}
              />
            </div>
            <ActionMenuItem
              label="Transcribe unpaired pages"
              meta={pageCountLabel(unpaired)}
              quietMeta
              disabled={busy || unpaired === 0}
              onSelect={() => void runTranscribe(close)}
            />
          </>
        )
      }
    </ActionMenu>
  );
}
