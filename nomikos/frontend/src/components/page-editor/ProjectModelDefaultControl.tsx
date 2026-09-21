import type { InferenceTask } from "../../api/client";

type ProjectModelDefaultControlProps = {
  projectId: string | undefined;
  task: InferenceTask;
  selectedModelId: string | null;
  defaultModelId: string | null;
  saving: boolean;
  saveError: string | null;
  onSave: (task: InferenceTask, modelId: string) => void;
};

function taskNoun(task: InferenceTask): string {
  return task === "transcribe" ? "transcription" : "segmentation";
}

/**
 * The explicit project-default action beside a model picker. When the
 * selected model already is the project default it reads as a quiet state
 * instead of an action. A failed save stays on the button's title so the
 * control never grows the toolbar, and the button stays enabled for retry.
 */
export function ProjectModelDefaultControl({
  projectId,
  task,
  selectedModelId,
  defaultModelId,
  saving,
  saveError,
  onSave,
}: ProjectModelDefaultControlProps) {
  if (!projectId || !selectedModelId) return null;
  const noun = taskNoun(task);
  if (defaultModelId === selectedModelId) {
    return (
      <span
        className="pe-model__label"
        title={`The default ${noun} model for new jobs in this project, for all members.`}
      >
        Project default
      </span>
    );
  }
  return (
    <button
      type="button"
      className="btn btn-outline btn-xs"
      disabled={saving}
      title={
        saveError ??
        `Set as the default ${noun} model for new jobs in this project, for all members.`
      }
      onClick={() => onSave(task, selectedModelId)}
    >
      {saving ? "Saving…" : "Set as project default"}
    </button>
  );
}
