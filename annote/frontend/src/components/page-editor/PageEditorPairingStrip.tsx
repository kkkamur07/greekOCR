import { useEffect, useState, type ChangeEventHandler } from 'react';
import type { LineResponse, TranscriptionLayerResponse } from '../../api/client';
import { PageEditorOcrReviewPane } from './PageEditorOcrReviewPane';
import {
  lineTranscriptionForLayer,
  showsModelSourceReview,
  transcriptionForOcrReview,
} from './hooks/utils';

type TextLine = { order: number; text: string; paired_line_id: string | null };

type DrawerTab = 'ocr' | 'truth' | 'pair';

type PageEditorPairingStripProps = {
  visible: boolean;
  editorMode: 'layout' | 'transcription';
  transcriptionLayers: TranscriptionLayerResponse[];
  selectedTranscriptionLayerId: string | null;
  onSelectTranscriptionLayer: ChangeEventHandler<HTMLSelectElement>;
  selectedSegmentNumber: number | null;
  selectedSegment: LineResponse | null;
  selectedTranscriptionLayer: TranscriptionLayerResponse | null;
  approvedTextDraft: string;
  onApprovedTextDraftChange: (value: string) => void;
  onSaveGroundTruthText: () => void;
  onPromoteSelectedSegmentToGroundTruth: () => void;
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
  return pairedIndex >= 0 ? `paired with Segment ${pairedIndex + 1}` : '';
}

function defaultTab(
  editorMode: 'layout' | 'transcription',
  hasPairingWork: boolean,
): DrawerTab {
  if (editorMode === 'transcription' || hasPairingWork) return 'pair';
  return 'ocr';
}

export function PageEditorPairingStrip({
  visible,
  editorMode,
  transcriptionLayers,
  selectedTranscriptionLayerId,
  onSelectTranscriptionLayer,
  selectedSegmentNumber,
  selectedSegment,
  selectedTranscriptionLayer,
  approvedTextDraft,
  onApprovedTextDraftChange,
  onSaveGroundTruthText,
  onPromoteSelectedSegmentToGroundTruth,
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
  const hasPairingWork = textLines.length > 0 || pairingProgress.total_lines > 0;
  const [activeTab, setActiveTab] = useState<DrawerTab>(() =>
    defaultTab(editorMode, hasPairingWork),
  );
  const [collapsed, setCollapsed] = useState(false);

  const groundTruthTranscription =
    selectedSegment && selectedTranscriptionLayer?.kind === 'ground_truth'
      ? lineTranscriptionForLayer(selectedSegment, selectedTranscriptionLayer.id)
      : null;
  const showGroundTruthEditor =
    Boolean(groundTruthTranscription) && !showsModelSourceReview(groundTruthTranscription);
  const ocrReviewTranscription =
    selectedSegment && selectedTranscriptionLayer
      ? transcriptionForOcrReview(selectedSegment, selectedTranscriptionLayer)
      : null;
  const showOcrReview = Boolean(ocrReviewTranscription);
  const showApprovedEditor = Boolean(selectedSegmentNumber) && !showGroundTruthEditor && !showOcrReview;

  useEffect(() => {
    if (!visible) return;
    if (editorMode === 'transcription') {
      setActiveTab('pair');
      return;
    }
    if (selectedSegmentId) {
      if (showGroundTruthEditor || showApprovedEditor) setActiveTab('truth');
      else if (showOcrReview) setActiveTab('ocr');
      else setActiveTab('ocr');
      setCollapsed(false);
      return;
    }
    if (hasPairingWork) setActiveTab('pair');
  }, [
    visible,
    editorMode,
    selectedSegmentId,
    showGroundTruthEditor,
    showApprovedEditor,
    showOcrReview,
    hasPairingWork,
  ]);

  if (!visible) {
    return null;
  }

  const drawerTitle = selectedSegmentNumber
    ? `Segment ${selectedSegmentNumber}`
    : hasPairingWork
      ? 'Page transcription'
      : 'Transcription';

  return (
    <div
      className={`pe-drawer ${collapsed ? 'pe-drawer--collapsed' : ''}`}
      aria-label="Segment pairing panel"
      role="complementary"
    >
      <div className="pe-drawer__bar">
        <span className="pe-drawer__bar-title">{drawerTitle}</span>
        {selectedTranscriptionLayer && (
          <span className="pe-drawer__bar-meta">· {selectedTranscriptionLayer.name}</span>
        )}
        <span className="pe-drawer__bar-spacer" />
        <button
          type="button"
          className="btn btn--ghost btn--sm"
          aria-expanded={!collapsed}
          onClick={() => setCollapsed((open) => !open)}
          title={collapsed ? 'Expand panel' : 'Collapse panel'}
        >
          {collapsed ? '▴' : '▾'}
        </button>
      </div>

      <div className="pe-drawer__tabs" role="tablist" aria-label="Transcription panels">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'ocr'}
          className={`pe-drawer__tab ${activeTab === 'ocr' ? 'pe-drawer__tab--active' : ''}`}
          onClick={() => setActiveTab('ocr')}
        >
          OCR review
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'truth'}
          className={`pe-drawer__tab ${activeTab === 'truth' ? 'pe-drawer__tab--active' : ''}`}
          onClick={() => setActiveTab('truth')}
        >
          Ground truth
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'pair'}
          className={`pe-drawer__tab ${activeTab === 'pair' ? 'pe-drawer__tab--active' : ''}`}
          onClick={() => setActiveTab('pair')}
        >
          Pairing
        </button>
      </div>

      <div className="pe-drawer__body">
        <div
          className={`pe-drawer__panel ${activeTab === 'ocr' ? 'pe-drawer__panel--active' : ''}`}
          role="tabpanel"
          aria-label="OCR output for selected segment"
        >
          {selectedSegmentNumber ? (
            <>
              <span className="visually-hidden">Selected Segment {selectedSegmentNumber}</span>
              <p className="pe-pairing__title">Segment {selectedSegmentNumber}</p>
            </>
          ) : (
            <p className="pe-pairing__muted">Select a segment on the canvas.</p>
          )}
          <label className="pe-pairing__label" style={{ marginTop: 4 }}>
            Layer
            <select
              className="pe-pairing__select"
              aria-label="Transcription layer"
              value={selectedTranscriptionLayerId ?? ''}
              onChange={onSelectTranscriptionLayer}
              style={{ marginTop: 4, maxWidth: 280 }}
            >
              {transcriptionLayers.map((layer) => (
                <option key={layer.id} value={layer.id}>
                  {layer.name}
                  {layer.kind === 'model' ? ' (read-only)' : ''}
                </option>
              ))}
            </select>
          </label>
          {selectedSegment && showOcrReview && (
            <>
              <PageEditorOcrReviewPane
                segmentNumber={selectedSegmentNumber}
                transcription={ocrReviewTranscription}
              />
              <div className="pe-pairing__btn-row">
                <button
                  type="button"
                  className="btn btn--accent btn--sm"
                  onClick={() => void onPromoteSelectedSegmentToGroundTruth()}
                >
                  Save
                </button>
              </div>
            </>
          )}
          {selectedSegment && !showOcrReview && !showGroundTruthEditor && !showApprovedEditor && (
            <p className="pe-pairing__muted">No OCR text on this layer yet.</p>
          )}
        </div>

        <div
          className={`pe-drawer__panel ${activeTab === 'truth' ? 'pe-drawer__panel--active' : ''}`}
          role="tabpanel"
          aria-label="Ground truth editor"
        >
          {selectedSegmentNumber ? (
            <>
              <span className="visually-hidden">Selected Segment {selectedSegmentNumber}</span>
              <p className="pe-pairing__title">Segment {selectedSegmentNumber}</p>
            </>
          ) : (
            <p className="pe-pairing__muted">Select a segment to edit.</p>
          )}
          {selectedSegment && showGroundTruthEditor && (
            <>
              <label className="pe-pairing__label">
                Edit or accept OCR output
                <textarea
                  className="pe-pairing__textarea"
                  aria-label="Ground truth text for selected Segment"
                  value={approvedTextDraft}
                  rows={3}
                  onChange={(event) => onApprovedTextDraftChange(event.target.value)}
                  style={{ marginTop: 4 }}
                />
              </label>
              <div className="pe-pairing__btn-row">
                <button
                  type="button"
                  className="btn btn--active btn--sm"
                  onClick={() => void onSaveGroundTruthText()}
                >
                  Save ground truth text
                </button>
              </div>
            </>
          )}
          {showApprovedEditor && (
            <>
              <span className="visually-hidden">Selected Segment {selectedSegmentNumber}</span>
              <label className="pe-pairing__label">
                Approved text
                <textarea
                  className="pe-pairing__textarea"
                  aria-label="Approved text for selected Segment"
                  value={approvedTextDraft}
                  rows={3}
                  onChange={(event) => onApprovedTextDraftChange(event.target.value)}
                  style={{ marginTop: 4 }}
                />
              </label>
              <div className="pe-pairing__btn-row">
                <button
                  type="button"
                  className="btn btn--active btn--sm"
                  onClick={() => void onSaveApprovedText()}
                >
                  Save approved text
                </button>
              </div>
            </>
          )}
          {selectedSegmentNumber && (
            <p className="pe-pairing__muted">
              Segment {selectedSegmentNumber} of {lines.length}
            </p>
          )}
        </div>

        <div
          className={`pe-drawer__panel ${activeTab === 'pair' ? 'pe-drawer__panel--active' : ''}`}
          role="tabpanel"
          aria-label="Text line pairing"
        >
          <p className="pe-pairing__title">Pair text lines</p>
          <label className="pe-pairing__label">
            Paste full page text
            <textarea
              className="pe-pairing__textarea"
              aria-label="Page transcription text"
              value={pageTranscriptionText}
              rows={2}
              onChange={(event) => onPageTranscriptionTextChange(event.target.value)}
              style={{ marginTop: 4 }}
            />
          </label>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => void onImportPageTranscription()}
          >
            Import page transcription
          </button>
          {textLines.map((textLine) => {
            const paired = Boolean(textLine.paired_line_id);
            const pairedLabel = pairedSegmentLabel(textLine, lines);
            const isSelectedPair = selectedSegmentId && !paired;
            return (
              <div
                key={textLine.order}
                className={`pe-tl-item ${isSelectedPair ? 'pe-tl-item--selected' : ''}`}
              >
                <div className="pe-tl-header">
                  <span className="pe-tl-num">Text line {textLine.order + 1}</span>
                  {paired && (
                    <span className="visually-hidden">
                      Text line {textLine.order + 1} · {pairedLabel}
                    </span>
                  )}
                  <span className={paired ? 'pe-tl-paired' : 'pe-tl-unpaired'}>
                    {paired ? pairedLabel : 'unpaired'}
                  </span>
                </div>
                <p className="pe-tl-text">{textLine.text}</p>
                <button
                  type="button"
                  className="btn btn--active btn--sm"
                  style={{ alignSelf: 'flex-start', fontSize: '0.75rem', padding: '4px 8px' }}
                  disabled={!selectedSegmentId}
                  onClick={() => void onPairTextLine(textLine.order)}
                >
                  Pair Text line {textLine.order + 1}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
