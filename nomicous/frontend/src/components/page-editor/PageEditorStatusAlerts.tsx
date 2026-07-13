import { useEffect } from "react";
import { toast } from "../ui/toast";

type PageEditorStatusAlertsProps = {
  saveMessage: string | null;
  transcriptionSaveMessage: string | null;
  ocrMessage: string | null;
  segmentMessage: string | null;
  mutationError: string | null;
  pairingError: string | null;
  layoutError: string | null;
  lineError: string | null;
};

function StatusItem({
  message,
  variant = "error",
}: {
  message: string;
  variant?: "error" | "warning";
}) {
  return (
    <div className={`pe-status-item pe-status-item--${variant}`}>
      {variant === "error" && <span aria-hidden="true">✕</span>}
      {variant === "warning" && <span aria-hidden="true">!</span>}
      <span>{message}</span>
    </div>
  );
}

/** Success/completion feedback uses auto-dismiss toasts; only errors stay sticky. */
export function PageEditorStatusAlerts({
  saveMessage,
  transcriptionSaveMessage,
  ocrMessage,
  segmentMessage,
  mutationError,
  pairingError,
  layoutError,
  lineError,
}: PageEditorStatusAlertsProps) {
  useEffect(() => {
    if (saveMessage) toast.success(saveMessage);
  }, [saveMessage]);
  useEffect(() => {
    if (transcriptionSaveMessage) toast.success(transcriptionSaveMessage);
  }, [transcriptionSaveMessage]);
  useEffect(() => {
    if (ocrMessage) toast.success(ocrMessage);
  }, [ocrMessage]);
  useEffect(() => {
    if (segmentMessage) toast.success(segmentMessage);
  }, [segmentMessage]);
  useEffect(() => {
    if (mutationError) toast.error(mutationError);
  }, [mutationError]);
  useEffect(() => {
    if (pairingError) toast.error(pairingError);
  }, [pairingError]);
  useEffect(() => {
    if (layoutError) toast.error(`Layout API unavailable: ${layoutError}`);
  }, [layoutError]);
  useEffect(() => {
    if (lineError) toast.error(`Segment API unavailable: ${lineError}`);
  }, [lineError]);

  const showSticky = mutationError || pairingError || layoutError || lineError;
  if (!showSticky) return null;

  return (
    <div className="pe-status-alerts">
      {mutationError && <StatusItem message={mutationError} variant="error" />}
      {pairingError && <StatusItem message={pairingError} variant="warning" />}
      {layoutError && (
        <StatusItem
          message={`Layout API unavailable: ${layoutError}`}
          variant="warning"
        />
      )}
      {lineError && (
        <StatusItem
          message={`Segment API unavailable: ${lineError}`}
          variant="warning"
        />
      )}
    </div>
  );
}

export function hasPageEditorStatusAlerts(
  props: PageEditorStatusAlertsProps,
): boolean {
  return Boolean(
    props.mutationError ||
    props.pairingError ||
    props.layoutError ||
    props.lineError,
  );
}
