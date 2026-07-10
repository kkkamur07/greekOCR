import { CharacterConfidenceText } from "./CharacterConfidenceText";
import {
  characterConfidencesForTranscription,
  type LineTranscriptionWithCharacterConfidence,
} from "./characterConfidence";

type ModelOutputBlockProps = {
  transcription: LineTranscriptionWithCharacterConfidence | null;
  segmentNumber: number | null;
  layout?: "stack" | "inline";
  onCopy?: () => void;
  onRerunOcr?: () => void;
  ocrRunning?: boolean;
  ocrDisabled?: boolean;
  ocrDisabledReason?: string;
};

export function ModelOutputBlock({
  transcription,
  segmentNumber,
  layout = "inline",
  onCopy,
  onRerunOcr,
  ocrRunning = false,
  ocrDisabled = false,
  ocrDisabledReason,
}: ModelOutputBlockProps) {
  const characterConfidences = transcription
    ? characterConfidencesForTranscription(transcription)
    : [];

  const handleCopy = async () => {
    if (!transcription?.text) return;
    await navigator.clipboard.writeText(transcription.text);
    onCopy?.();
  };

  const hasTranscription = transcription !== null;
  const hasText = Boolean(transcription?.text?.trim());

  const emptyMessage = !segmentNumber
    ? "Select a segment to view model output."
    : hasTranscription && !hasText
      ? "OCR finished with no text for this segment."
      : "No OCR yet. Run OCR on this segment.";
  const ocrButtonLabel = hasText
    ? ocrRunning
      ? "Running…"
      : "Re-run OCR"
    : ocrRunning
      ? "Running…"
      : "Run OCR";

  const ariaLabel = segmentNumber
    ? `OCR model output for segment ${segmentNumber}`
    : "OCR model output";

  if (layout === "inline") {
    return (
      <div className="pe-model-inline">
        <span className="pe-model-inline__label" id="model-output-label">
          Model output:
        </span>
        <div
          className="pe-model-inline__body"
          role="text"
          aria-labelledby="model-output-label"
          aria-describedby="confidence-legend"
        >
          {hasText ? (
            <CharacterConfidenceText
              characterConfidences={characterConfidences}
              ariaLabel={ariaLabel}
            />
          ) : (
            <span className="pe-model-inline__empty">{emptyMessage}</span>
          )}
        </div>
        <div className="pe-model-inline__actions">
          {hasText && (
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => void handleCopy()}
            >
              Copy
            </button>
          )}
          {onRerunOcr && (
            <button
              type="button"
              className="btn btn-outline btn-xs"
              disabled={ocrDisabled || ocrRunning}
              title={ocrDisabled ? ocrDisabledReason : undefined}
              onClick={() => void onRerunOcr()}
            >
              {ocrButtonLabel}
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="pe-tx-block pe-tx-block--model">
      <div className="pe-tx-block__head">
        <span className="pe-tx-block__label" id="model-output-label">
          Model output
        </span>
        {hasText && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={() => void handleCopy()}
          >
            Copy
          </button>
        )}
      </div>
      <div
        className="pe-model-preview"
        role="text"
        aria-labelledby="model-output-label"
        aria-describedby="confidence-legend"
      >
        {hasText ? (
          <CharacterConfidenceText
            characterConfidences={characterConfidences}
            ariaLabel={ariaLabel}
          />
        ) : (
          <span className="text-muted text-sm">{emptyMessage}</span>
        )}
      </div>
    </div>
  );
}
