import type { InferenceModelResponse } from "../../api/client";

type PageEditorModelSelectProps = {
  transcribeModels: InferenceModelResponse[];
  selectedTranscribeModelId: string | null;
  onSelectedTranscribeModelIdChange: (modelId: string | null) => void;
  disabled?: boolean;
};

export function PageEditorModelSelect({
  transcribeModels,
  selectedTranscribeModelId,
  onSelectedTranscribeModelIdChange,
  disabled = false,
}: PageEditorModelSelectProps) {
  return (
    <label className="pe-model">
      <span className="pe-model__label">HTR</span>
      <select
        className="pe-model__select"
        aria-label="HTR transcription model"
        value={selectedTranscribeModelId ?? ""}
        disabled={disabled || transcribeModels.length === 0}
        onChange={(event) =>
          onSelectedTranscribeModelIdChange(event.target.value || null)
        }
      >
        {transcribeModels.length === 0 ? (
          <option value="">No models</option>
        ) : (
          transcribeModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))
        )}
      </select>
    </label>
  );
}
