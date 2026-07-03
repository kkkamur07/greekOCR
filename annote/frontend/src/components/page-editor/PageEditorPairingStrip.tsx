import { Button, Input, Space, Typography } from 'antd';
import type { ChangeEventHandler } from 'react';
import type { LineResponse, TranscriptionLayerResponse } from '../../api/client';
import { PageEditorOcrReviewPane } from './PageEditorOcrReviewPane';
import { lineTranscriptionForLayer } from './hooks/utils';

type TextLine = { order: number; text: string; paired_line_id: string | null };

type PageEditorPairingStripProps = {
  visible: boolean;
  transcriptionLayers: TranscriptionLayerResponse[];
  selectedTranscriptionLayerId: string | null;
  onSelectTranscriptionLayer: ChangeEventHandler<HTMLSelectElement>;
  selectedSegmentNumber: number | null;
  selectedSegment: LineResponse | null;
  selectedTranscriptionLayer: TranscriptionLayerResponse | null;
  approvedTextDraft: string;
  onApprovedTextDraftChange: (value: string) => void;
  lineTextForLayer: (line: LineResponse, layerId: string) => string;
  onSaveGroundTruthText: () => void;
  onCopySelectedLayerToGroundTruth: (lineIds: string[] | null) => void;
  pairingProgress: { paired_lines: number; total_lines: number; percent: number };
  pageTranscriptionText: string;
  onPageTranscriptionTextChange: (value: string) => void;
  onImportPageTranscription: () => void;
  textLines: TextLine[];
  lines: LineResponse[];
  selectedSegmentId: string | null;
  onPairTextLine: (order: number) => void;
  onSaveApprovedText: () => void;
};

function pairedSegmentLabel(textLine: TextLine, lines: LineResponse[]): string {
  const pairedIndex = textLine.paired_line_id
    ? [...lines]
        .sort((a, b) => a.order - b.order)
        .findIndex((line) => line.id === textLine.paired_line_id)
    : -1;
  return pairedIndex >= 0 ? ` · paired with Segment ${pairedIndex + 1}` : '';
}

export function PageEditorPairingStrip({
  visible,
  transcriptionLayers,
  selectedTranscriptionLayerId,
  onSelectTranscriptionLayer,
  selectedSegmentNumber,
  selectedSegment,
  selectedTranscriptionLayer,
  approvedTextDraft,
  onApprovedTextDraftChange,
  lineTextForLayer,
  onSaveGroundTruthText,
  onCopySelectedLayerToGroundTruth,
  pairingProgress,
  pageTranscriptionText,
  onPageTranscriptionTextChange,
  onImportPageTranscription,
  textLines,
  lines,
  selectedSegmentId,
  onPairTextLine,
  onSaveApprovedText,
}: PageEditorPairingStripProps) {
  if (!visible) {
    return null;
  }

  return (
    <div
      style={{
        flexShrink: 0,
        borderTop: '1px solid #e5e7eb',
        background: '#fff',
        padding: 12,
      }}
    >
      <div style={{ display: 'grid', gap: 12 }}>
        <label style={{ display: 'grid', gap: 8, color: '#c5ccd6' }}>
          Transcription layer
          <select
            aria-label="Transcription layer"
            value={selectedTranscriptionLayerId ?? ''}
            onChange={onSelectTranscriptionLayer}
          >
            {transcriptionLayers.map((layer) => (
              <option key={layer.id} value={layer.id}>
                {layer.name}
                {layer.kind === 'model' ? ' (read-only)' : ''}
              </option>
            ))}
          </select>
        </label>
        <Typography.Text>
          {selectedSegmentNumber
            ? `Selected Segment ${selectedSegmentNumber}`
            : 'Select a Segment to view transcription text.'}
        </Typography.Text>
        {selectedSegment && selectedTranscriptionLayer?.kind === 'ground_truth' && (
          <>
            <label style={{ display: 'grid', gap: 8 }}>
              Ground truth text for selected Segment
              <Input.TextArea
                aria-label="Ground truth text for selected Segment"
                value={approvedTextDraft}
                rows={3}
                onChange={(event) => onApprovedTextDraftChange(event.target.value)}
              />
            </label>
            <Button type="primary" onClick={() => void onSaveGroundTruthText()}>
              Save Ground truth text
            </Button>
          </>
        )}
        {selectedSegment && selectedTranscriptionLayer?.kind === 'model' && (
          <>
            <PageEditorOcrReviewPane
              segmentNumber={selectedSegmentNumber}
              transcription={lineTranscriptionForLayer(selectedSegment, selectedTranscriptionLayer.id)}
            />
            <Space wrap>
              <Button
                type="primary"
                onClick={() => void onCopySelectedLayerToGroundTruth([selectedSegment.id])}
              >
                Copy selected Segment to Ground truth
              </Button>
              <Button onClick={() => void onCopySelectedLayerToGroundTruth(null)}>
                Copy whole Page to Ground truth
              </Button>
            </Space>
          </>
        )}
        <details>
          <summary>Page transcription and pairing</summary>
          <div style={{ display: 'grid', gap: 8, paddingTop: 8 }}>
            <Typography.Text>
              Pairing progress: {pairingProgress.paired_lines}/{pairingProgress.total_lines} Lines
              paired
            </Typography.Text>
            <label style={{ display: 'grid', gap: 8 }}>
              Page transcription text
              <Input.TextArea
                aria-label="Page transcription text"
                value={pageTranscriptionText}
                rows={4}
                onChange={(event) => onPageTranscriptionTextChange(event.target.value)}
              />
            </label>
            <Button onClick={() => void onImportPageTranscription()}>Import page transcription</Button>
          </div>
        </details>
        {textLines.map((textLine) => (
          <div
            key={textLine.order}
            style={{
              border: '1px solid #3b4350',
              borderRadius: 6,
              padding: 8,
            }}
          >
            <Typography.Text style={{ color: '#d8c7a1' }}>
              Text line {textLine.order + 1}
              {pairedSegmentLabel(textLine, lines)}
            </Typography.Text>
            <Typography.Paragraph style={{ marginBottom: 8 }}>{textLine.text}</Typography.Paragraph>
            <Button
              disabled={!selectedSegmentId}
              onClick={() => void onPairTextLine(textLine.order)}
            >
              Pair Text line {textLine.order + 1}
            </Button>
          </div>
        ))}
        {selectedSegmentNumber && (
          <div style={{ display: 'grid', gap: 8 }}>
            <Typography.Text>Selected Segment {selectedSegmentNumber}</Typography.Text>
            <label style={{ display: 'grid', gap: 8 }}>
              Approved text for selected Segment
              <Input.TextArea
                aria-label="Approved text for selected Segment"
                value={approvedTextDraft}
                rows={3}
                onChange={(event) => onApprovedTextDraftChange(event.target.value)}
              />
            </label>
            <Button type="primary" onClick={() => void onSaveApprovedText()}>
              Save approved text
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
