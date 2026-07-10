import { modelDisplayName } from "../../inference/modelDisplayName";

type PageEditorLocalInferenceBannerProps = {
  registryModelId: string | null;
  onUseCloudInstead: () => void;
};

export function PageEditorLocalInferenceBanner({
  registryModelId,
  onUseCloudInstead,
}: PageEditorLocalInferenceBannerProps) {
  if (!registryModelId) {
    return null;
  }

  return (
    <div
      className="pe-inference-banner pe-inference-banner--active"
      role="status"
      aria-live="polite"
    >
      <span>
        Downloading the <strong>{modelDisplayName(registryModelId)}</strong> on
        this machine…
      </span>
      <button
        type="button"
        className="pe-inference-banner__action"
        onClick={onUseCloudInstead}
      >
        Use cloud instead
      </button>
    </div>
  );
}
