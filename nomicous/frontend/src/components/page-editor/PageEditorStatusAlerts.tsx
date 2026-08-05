import { useEffect } from "react";
import { toast } from "../ui/toast";

type PageEditorStatusAlertsProps = {
  /**
   * "No inference host had capacity", as the platform explained it when it
   * refused the submission. Deliberately not routed through a toast: it names
   * something the researcher can fix, so it stays on screen until the next run.
   */
  submissionRefusal: string | null;
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
  submissionRefusal,
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

  const showSticky =
    submissionRefusal ||
    mutationError ||
    pairingError ||
    layoutError ||
    lineError;
  if (!showSticky) return null;

  return (
    <div className="pe-status-alerts">
      {submissionRefusal && (
        <StatusItem message={submissionRefusal} variant="warning" />
      )}
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
    props.submissionRefusal ||
    props.mutationError ||
    props.pairingError ||
    props.layoutError ||
    props.lineError,
  );
}
