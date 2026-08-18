type PageEditorInferenceStatusProps = {
  loading: boolean;
  /** **Capacity** for this account's own computer, as the platform reports it. */
  hasLocalCapacity: boolean;
  /** The account-level **host preference**: "use my computer when available". */
  preferLocalInference: boolean;
};

type StatusVariant = "checking" | "connected" | "cloud" | "unavailable";

function resolveVariant({
  loading,
  hasLocalCapacity,
  preferLocalInference,
}: PageEditorInferenceStatusProps): StatusVariant {
  if (!preferLocalInference) return "cloud";
  if (loading) return "checking";
  if (hasLocalCapacity) return "connected";
  return "unavailable";
}

const LABELS: Record<StatusVariant, string> = {
  checking: "checking…",
  connected: "ready",
  cloud: "using cloud",
  unavailable: "not running",
};

/**
 * These read as **capacity**, not as a promise. Nothing here claims where a
 * given job ran - the job says that itself, which is the whole point of the
 * announcement line.
 */
const TITLES: Record<StatusVariant, string> = {
  checking: "Reading whether the nomikos agent is running on this computer…",
  connected:
    "The nomikos agent is running on this computer, so jobs can run here.",
  cloud:
    "This account has not asked for its own computer, so jobs run in the cloud.",
  unavailable:
    "The nomikos agent is not running on this computer, so jobs go to the cloud.",
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
