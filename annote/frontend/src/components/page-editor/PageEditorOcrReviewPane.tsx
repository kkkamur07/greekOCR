import { Progress, Typography } from 'antd';
import type { LineTranscriptionResponse } from '../../api/client';
import { CharacterConfidenceText } from './CharacterConfidenceText';
import {
  characterConfidencesForTranscription,
  confidenceLabelColor,
  formatConfidencePercent,
  hasDistinctCharacterConfidences,
  type LineTranscriptionWithCharacterConfidence,
} from './characterConfidence';

type PageEditorOcrReviewPaneProps = {
  segmentNumber: number | null;
  transcription: LineTranscriptionResponse | null;
};

export function PageEditorOcrReviewPane({
  segmentNumber,
  transcription,
}: PageEditorOcrReviewPaneProps) {
  if (!transcription) {
    return (
      <Typography.Text type="secondary">
        {segmentNumber
          ? `Segment ${segmentNumber} has no OCR text on this layer yet.`
          : 'Select a Segment to review OCR output.'}
      </Typography.Text>
    );
  }

  const enriched = transcription as LineTranscriptionWithCharacterConfidence;
  const characterConfidences = characterConfidencesForTranscription(enriched);
  const lineConfidence = transcription.confidence;
  const distinctScores = hasDistinctCharacterConfidences(enriched);

  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <Typography.Text>
        {segmentNumber ? `OCR review · Segment ${segmentNumber}` : 'OCR review'}
      </Typography.Text>
      {lineConfidence !== null && (
        <div aria-label="Line confidence">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <Typography.Text type="secondary">Line confidence</Typography.Text>
            <Typography.Text strong style={{ color: confidenceLabelColor(lineConfidence) }}>
              {formatConfidencePercent(lineConfidence)}
            </Typography.Text>
          </div>
          <Progress
            percent={lineConfidence * 100}
            showInfo={false}
            strokeColor={confidenceLabelColor(lineConfidence)}
            size="small"
          />
        </div>
      )}
      {!distinctScores && lineConfidence !== null && transcription.text.length > 0 && (
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          Per-character scores are not available yet; characters use the line confidence color.
        </Typography.Text>
      )}
      {transcription.text.length > 0 ? (
        <CharacterConfidenceText
          characterConfidences={characterConfidences}
          ariaLabel="OCR text with per-character confidence highlighting"
        />
      ) : (
        <Typography.Text type="secondary">No OCR text for this Segment.</Typography.Text>
      )}
    </div>
  );
}
