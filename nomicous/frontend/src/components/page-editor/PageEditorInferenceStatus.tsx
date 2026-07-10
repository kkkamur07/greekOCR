type PageEditorInferenceStatusProps = {
  probing: boolean;
  helperAvailable: boolean;
  preferCloud: boolean;
};

type StatusVariant = "checking" | "connected" | "cloud" | "unavailable";

function resolveVariant({
  probing,
  helperAvailable,
  preferCloud,
}: PageEditorInferenceStatusProps): StatusVariant {
  if (probing) return "checking";
  if (preferCloud) return "cloud";
  if (helperAvailable) return "connected";
  return "unavailable";
}

const LABELS: Record<StatusVariant, string> = {
  checking: "checking…",
  connected: "connected",
  cloud: "using cloud",
  unavailable: "not installed",
};

const TITLES: Record<StatusVariant, string> = {
  checking: "Looking for the Nomicous Inference Helper on this machine…",
  connected:
    "Local inference helper is running on 127.0.0.1:8001. OCR and segmentation run on your CPU.",
  cloud: "Cloud inference is selected. Jobs run on the server.",
  unavailable:
    "No local helper detected. Install it to run OCR and segmentation on your CPU.",
};

export function PageEditorInferenceStatus(
  props: PageEditorInferenceStatusProps,
) {
  const variant = resolveVariant(props);
  return (
    <div
      className={`pe-infstat pe-infstat--${variant}`}
      role="status"
      aria-live="polite"
      title={TITLES[variant]}
    >
      <span className="pe-infstat__dot" aria-hidden="true" />
      <span className="pe-infstat__label">
        {variant === "cloud" ? "Cloud inference" : "Local inference"}
        <span className="pe-infstat__sep"> · </span>
        <span className="pe-infstat__state">{LABELS[variant]}</span>
      </span>
    </div>
  );
}
