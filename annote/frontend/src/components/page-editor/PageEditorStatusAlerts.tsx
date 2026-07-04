import { Alert } from 'antd';

type PageEditorStatusAlertsProps = {
  saveMessage: string | null;
  transcriptionSaveMessage: string | null;
  ocrMessage: string | null;
  segmentMessage: string | null;
  mutationError: string | null;
  pairingError: string | null;
  reviewError: string | null;
  layoutError: string | null;
  lineError: string | null;
};

export function PageEditorStatusAlerts({
  saveMessage,
  transcriptionSaveMessage,
  ocrMessage,
  segmentMessage,
  mutationError,
  pairingError,
  reviewError,
  layoutError,
  lineError,
}: PageEditorStatusAlertsProps) {
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      {saveMessage && <Alert type="success" showIcon message={saveMessage} />}
      {transcriptionSaveMessage && (
        <Alert type="success" showIcon message={transcriptionSaveMessage} />
      )}
      {ocrMessage && <Alert type="success" showIcon message={ocrMessage} />}
      {segmentMessage && <Alert type="success" showIcon message={segmentMessage} />}
      {mutationError && <Alert type="error" showIcon message={mutationError} />}
      {pairingError && <Alert type="warning" showIcon message={pairingError} />}
      {reviewError && <Alert type="warning" showIcon message={reviewError} />}
      {layoutError && (
        <Alert type="warning" showIcon message="Layout API unavailable" description={layoutError} />
      )}
      {lineError && (
        <Alert type="warning" showIcon message="Segment API unavailable" description={lineError} />
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
      props.reviewError ||
      props.layoutError ||
      props.lineError,
  );
}
