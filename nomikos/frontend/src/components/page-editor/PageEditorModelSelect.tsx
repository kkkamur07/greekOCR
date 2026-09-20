import type { InferenceModelResponse } from "../../api/client";

type PageEditorModelSelectProps = {
  /** Short visible label, "HTR" or "Seg". */
  label: string;
  /** Accessible name for the select, "HTR transcription model" or similar. */
  ariaLabel: string;
  models: InferenceModelResponse[];
  selectedModelId: string | null;
  onSelectedModelIdChange: (modelId: string | null) => void;
  disabled?: boolean;
  /**
   * When true, a leading "Default" option (value null) is offered and the
   * select stays enabled even with no models. The HTR picker leaves this off,
   * so its rendered output is unchanged.
   */
  includeDefaultOption?: boolean;
};

export function PageEditorModelSelect({
  label,
  ariaLabel,
  models,
  selectedModelId,
  onSelectedModelIdChange,
  disabled = false,
  includeDefaultOption = false,
}: PageEditorModelSelectProps) {
  return (
    <label className="pe-model">
      <span className="pe-model__label">{label}</span>
      <select
        className="pe-model__select"
        aria-label={ariaLabel}
        value={selectedModelId ?? ""}
        disabled={disabled || (models.length === 0 && !includeDefaultOption)}
        onChange={(event) =>
          onSelectedModelIdChange(event.target.value || null)
        }
      >
        {models.length === 0 && !includeDefaultOption ? (
          <option value="">No models</option>
        ) : (
          <>
            {includeDefaultOption && <option value="">Default</option>}
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </>
        )}
      </select>
    </label>
  );
}
