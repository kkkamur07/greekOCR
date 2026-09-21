import { useEffect, useState } from "react";
import { api, type InferenceModelResponse } from "../../api/client";
import { useProjectModelDefaults } from "../page-editor/projectModelDefaults";
import { toast } from "../ui/toast";

type ProjectDefaultModelsPanelProps = {
  projectId: string;
};

const NO_DEFAULT = "";

/** The two tasks a project binds a model for. Binarization has no picker. */
type DefaultTask = "segment" | "transcribe";

const TASK_NOUN: Record<DefaultTask, string> = {
  segment: "segmentation",
  transcribe: "transcription",
};

/**
 * The one place a project's default models are seen and set.
 *
 * Every picker in the app starts on these bindings, so the setting belongs
 * where the project itself is configured rather than beside each run button,
 * where it competed with the action and told each researcher a different
 * story about what "default" meant.
 */
export function ProjectDefaultModelsPanel({
  projectId,
}: ProjectDefaultModelsPanelProps) {
  const [segmentModels, setSegmentModels] = useState<InferenceModelResponse[]>(
    [],
  );
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  const defaults = useProjectModelDefaults(projectId);

  // A failed catalog read leaves the rows empty and disabled rather than an
  // error banner: nothing here is lost by trying again later.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const catalog = await api.listInferenceModels();
        if (cancelled) return;
        setSegmentModels(catalog.filter((model) => model.task === "segment"));
        setTranscribeModels(
          catalog.filter((model) => model.task === "transcribe"),
        );
      } catch {
        if (cancelled) return;
        setSegmentModels([]);
        setTranscribeModels([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  async function handleChange(
    task: DefaultTask,
    models: InferenceModelResponse[],
    value: string,
  ) {
    const noun = TASK_NOUN[task];
    if (value === NO_DEFAULT) {
      const cleared = await defaults.clearDefault(task);
      if (cleared.ok) toast.success(`Default ${noun} model cleared`);
      else toast.error(cleared.message);
      return;
    }
    const saved = await defaults.saveDefault(task, value);
    if (saved.ok) {
      const name = models.find((model) => model.id === value)?.name ?? value;
      toast.success(`Default ${noun} model set to ${name}`);
    } else {
      toast.error(saved.message);
    }
  }

  function row(
    task: DefaultTask,
    label: string,
    models: InferenceModelResponse[],
  ) {
    const saving = defaults.saving === task;
    const inputId = `project-default-${task}`;
    return (
      <div className="default-models__row">
        <label className="default-models__label" htmlFor={inputId}>
          {label}
        </label>
        <div className="default-models__control">
          <select
            id={inputId}
            className="default-models__select"
            // The select always shows the stored binding, so a failed save
            // snaps back to what the project actually has.
            value={defaults.defaultModelId(task) ?? NO_DEFAULT}
            disabled={saving || models.length === 0}
            onChange={(event) =>
              void handleChange(task, models, event.target.value)
            }
          >
            <option value={NO_DEFAULT}>No default</option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          {saving && (
            <span className="default-models__saving" role="status">
              Saving…
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="entity-panel__section">
      <h2 className="entity-panel__heading">Default models</h2>
      <p className="entity-panel__hint">
        New segmentation and transcription jobs in this project start with these
        models. Anyone can still pick another model for a single run.
      </p>
      {row("segment", "Segmentation", segmentModels)}
      {row("transcribe", "Transcription", transcribeModels)}
    </div>
  );
}
