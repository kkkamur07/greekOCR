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
  variant = 'success',
}: {
  message: string;
  variant?: 'success' | 'error' | 'warning';
}) {
  return (
    <div className={`pe-status-item pe-status-item--${variant}`}>
      {variant === 'success' && <span aria-hidden="true">✓</span>}
      {variant === 'error' && <span aria-hidden="true">✕</span>}
      {variant === 'warning' && <span aria-hidden="true">!</span>}
      <span>{message}</span>
    </div>
  );
}

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
  return (
    <div className="pe-status-alerts">
      {saveMessage && <StatusItem message={saveMessage} />}
      {transcriptionSaveMessage && <StatusItem message={transcriptionSaveMessage} />}
      {ocrMessage && <StatusItem message={ocrMessage} />}
      {segmentMessage && <StatusItem message={segmentMessage} />}
      {mutationError && <StatusItem message={mutationError} variant="error" />}
      {pairingError && <StatusItem message={pairingError} variant="warning" />}
      {layoutError && (
        <StatusItem message={`Layout API unavailable: ${layoutError}`} variant="warning" />
      )}
      {lineError && (
        <StatusItem message={`Segment API unavailable: ${lineError}`} variant="warning" />
      )}
    </div>
  );
}

export function hasPageEditorStatusAlerts(props: PageEditorStatusAlertsProps): boolean {
  return Boolean(
    props.saveMessage ||
      props.transcriptionSaveMessage ||
      props.ocrMessage ||
      props.segmentMessage ||
      props.mutationError ||
      props.pairingError ||
      props.layoutError ||
      props.lineError,
  );
}
