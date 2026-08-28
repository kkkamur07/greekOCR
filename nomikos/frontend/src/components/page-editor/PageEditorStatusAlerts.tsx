import { useEffect } from "react";
import { toast } from "../ui/toast";
import type { StatusMessage } from "./statusMessage";

/**
 * One toast per message *raised*, not per distinct sentence.
 *
 * The effect is keyed on the message's token rather than its text, because
 * several of these sentences are constants and two saves in a row would
 * otherwise be one dependency that never changed. See `statusMessage`.
 */
function useSuccessToast(message: StatusMessage | null) {
  useEffect(() => {
    if (message) toast.success(message.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the token is the identity of this message
  }, [message?.at]);
}

type PageEditorStatusAlertsProps = {
  /**
   * "No inference host had capacity", as the platform explained it when it
   * refused the submission. Deliberately not routed through a toast: it names
   * something the researcher can fix, so it stays on screen until the next run.
   */
  submissionRefusal: string | null;
  saveMessage: StatusMessage | null;
  transcriptionSaveMessage: StatusMessage | null;
  ocrMessage: StatusMessage | null;
  segmentMessage: StatusMessage | null;
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
  useSuccessToast(saveMessage);
  useSuccessToast(transcriptionSaveMessage);
  useSuccessToast(ocrMessage);
  useSuccessToast(segmentMessage);
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
