import { type CSSProperties } from 'react';
import { Link } from 'react-router-dom';
import type { DocumentWithPartsResponse, InferenceModelResponse, LineResponse } from '../../api/client';
import { editorButton } from './editorButton';

const toolbarClusterStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  paddingInline: 4,
  borderLeft: '1px solid #e5e7eb',
} satisfies CSSProperties;

type TextLine = { order: number; text: string; paired_line_id: string | null };

type PageEditorToolbarProps = {
  projectId: string | undefined;
  documentId: string | undefined;
  document: DocumentWithPartsResponse;
  part: DocumentWithPartsResponse['parts'][number];
  partIndex: number;
  editorMode: 'layout' | 'transcription';
  onEditorModeChange: (mode: 'layout' | 'transcription') => void;
  drawMode: 'none' | 'rectangle' | 'polygon';
  onPickDrawMode: (mode: 'rectangle' | 'polygon') => void;
  onPanSelect: () => void;
  lines: LineResponse[];
  pairingProgress: { paired_lines: number; total_lines: number; percent: number };
  selectedSegmentId: string | null;
  selectedSegment: LineResponse | null;
  selectedLineId: string | null;
  pageTranscriptionText: string;
  onPageTranscriptionTextChange: (value: string) => void;
  onImportPageTranscription: () => void;
  textLines: TextLine[];
  onPairTextLine: (order: number) => void;
  onMoveSelectedSegmentRight: () => void;
  onDeleteSelectedSegment: () => void;
  onResetSelectedLine: () => void;
  actionsOpen: boolean;
  onActionsOpenChange: (open: boolean) => void;
  useOtsuRefinement: boolean;
  onUseOtsuRefinementChange: (value: boolean) => void;
  segmenting: boolean;
  ocrRunning: boolean;
  transcribeModels: InferenceModelResponse[];
  selectedTranscribeModelId: string | null;
  onSelectedTranscribeModelIdChange: (modelId: string | null) => void;
  onRunAutoSegment: () => void;
  onRunSegmentOcr: () => void;
  onRunPageOcr: () => void;
  onUpdateReviewStatus: (reviewed: boolean) => void;
};

function pairedSegmentLabel(textLine: TextLine, lines: LineResponse[]): string {
  const pairedIndex = textLine.paired_line_id
    ? [...lines]
        .sort((a, b) => a.order - b.order)
        .findIndex((line) => line.id === textLine.paired_line_id)
    : -1;
  return pairedIndex >= 0 ? ` · paired with Segment ${pairedIndex + 1}` : '';
}

export function PageEditorToolbar({
  projectId,
  documentId,
  document,
  part,
  partIndex,
  editorMode,
  onEditorModeChange,
  drawMode,
  onPickDrawMode,
  onPanSelect,
  lines,
  pairingProgress,
  selectedSegmentId,
  selectedSegment,
  selectedLineId,
  pageTranscriptionText,
  onPageTranscriptionTextChange,
  onImportPageTranscription,
  textLines,
  onPairTextLine,
  onMoveSelectedSegmentRight,
  onDeleteSelectedSegment,
  onResetSelectedLine,
  actionsOpen,
  onActionsOpenChange,
  useOtsuRefinement,
  onUseOtsuRefinementChange,
  segmenting,
  ocrRunning,
  transcribeModels,
  selectedTranscribeModelId,
  onSelectedTranscribeModelIdChange,
  onRunAutoSegment,
  onRunSegmentOcr,
  onRunPageOcr,
  onUpdateReviewStatus,
}: PageEditorToolbarProps) {
  return (
    <header
      style={{
        display: 'flex',
        flexShrink: 0,
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        borderBottom: '1px solid #e5e7eb',
        padding: '8px 12px',
      }}
    >
      <span className="visually-hidden">ANNOTE PAGE WORKSPACE</span>
      <h2 className="visually-hidden">{editorMode === 'layout' ? 'Layout edit' : 'Transcription edit'}</h2>
      <span className="visually-hidden">
        Review status: {part.reviewed ? 'Reviewed' : 'Unreviewed'}
      </span>
      {!selectedSegment && editorMode !== 'transcription' && (
        <span className="visually-hidden">
          Pairing progress: {pairingProgress.paired_lines}/{pairingProgress.total_lines} Lines paired
        </span>
      )}
      <span className="visually-hidden">
        {lines.length} {lines.length === 1 ? 'Segment' : 'Segments'}
      </span>
      <div className="visually-hidden">
        <label>
          Page transcription text
          <textarea
            aria-label="Page transcription text"
            value={pageTranscriptionText}
            onChange={(event) => onPageTranscriptionTextChange(event.target.value)}
          />
        </label>
        <button type="button" onClick={() => void onImportPageTranscription()}>
          Import page transcription
        </button>
        {!selectedSegmentId &&
          textLines.map((textLine) => (
            <div key={textLine.order}>
              <span>
                Text line {textLine.order + 1}
                {pairedSegmentLabel(textLine, lines)}
              </span>
              <button
                type="button"
                disabled={!selectedSegmentId}
                onClick={() => void onPairTextLine(textLine.order)}
              >
                Pair Text line {textLine.order + 1}
              </button>
            </div>
          ))}
      </div>
      <div style={{ display: 'flex', minWidth: 0, alignItems: 'center', gap: 8 }}>
        {projectId && documentId && (
          <Link
            to={`/projects/${projectId}/documents/${documentId}`}
            aria-label="Document parts"
            style={{ flexShrink: 0, color: '#6b7280', fontSize: 14 }}
          >
            ← Back
          </Link>
        )}
        <h1
          title={`${document.name} · Page ${partIndex}`}
          style={{
            margin: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            fontSize: 16,
            fontWeight: 500,
          }}
        >
          {document.name} · Page {partIndex}
        </h1>
        <span style={{ color: '#92400e', fontSize: 12 }}>
          {part.reviewed ? 'reviewed' : 'unreviewed'}
        </span>
      </div>

      <div style={{ display: 'flex', minWidth: 0, flex: 1, justifyContent: 'center', paddingInline: 8 }}>
        <div style={{ color: '#6b7280', fontSize: 13 }}>
          Pairing {pairingProgress.paired_lines}/{pairingProgress.total_lines} · {lines.length}{' '}
          {lines.length === 1 ? 'segment' : 'segments'}
        </div>
      </div>

      <div style={{ display: 'flex', flexShrink: 0, alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
        <button type="button" onClick={onPanSelect} className={editorButton(drawMode === 'none' && editorMode === 'layout')}>
          Pan
        </button>
        <button type="button" onClick={onPanSelect} className={editorButton(drawMode === 'none' && editorMode === 'layout')}>
          Select
        </button>
        <button
          type="button"
          aria-label="Rectangle segment"
          onClick={() => onPickDrawMode('rectangle')}
          className={editorButton(drawMode === 'rectangle')}
        >
          Rect
        </button>
        <button
          type="button"
          aria-label="Polygon segment"
          onClick={() => onPickDrawMode('polygon')}
          className={editorButton(drawMode === 'polygon')}
        >
          Poly
        </button>
        <button
          type="button"
          aria-label="Transcription edit"
          onClick={() => onEditorModeChange(editorMode === 'transcription' ? 'layout' : 'transcription')}
          className={editorButton(editorMode === 'transcription')}
        >
          Text
        </button>
        <div style={toolbarClusterStyle}>
          <button
            type="button"
            aria-label="Move segment right"
            disabled={!selectedSegmentId}
            onClick={() => void onMoveSelectedSegmentRight()}
            className={editorButton(false)}
          >
            Move
          </button>
          <button
            type="button"
            aria-label={selectedSegmentId ? 'Delete Segment' : 'Delete'}
            disabled={!selectedSegmentId && !selectedLineId}
            onClick={() => {
              if (selectedSegmentId) void onDeleteSelectedSegment();
              if (selectedLineId) void onResetSelectedLine();
            }}
            className="btn btn--danger-ghost btn--sm"
          >
            Del
          </button>
        </div>
        <div style={toolbarClusterStyle}>
          <button type="button" className={editorButton(true)}>
            Lines
          </button>
          <button type="button" disabled className={editorButton(false)}>
            Ceiling
          </button>
        </div>
        <div style={{ ...toolbarClusterStyle, position: 'relative' }}>
          <button
            type="button"
            aria-haspopup="menu"
            aria-expanded={actionsOpen}
            onClick={() => onActionsOpenChange(!actionsOpen)}
            className="btn btn--ghost btn--sm"
          >
            Process
          </button>
          {actionsOpen && (
            <div
              role="menu"
              aria-label="Page processing actions"
              style={{
                position: 'absolute',
                top: 'calc(100% + 6px)',
                right: 0,
                zIndex: 20,
                display: 'grid',
                gap: 6,
                width: 240,
                padding: 8,
                border: '1px solid #d1d5db',
                borderRadius: 8,
                background: '#fff',
                boxShadow: '0 12px 32px rgba(15, 23, 42, 0.18)',
              }}
            >
              <div
                style={{
                  padding: '2px 6px',
                  fontSize: 11,
                  fontWeight: 600,
                  color: '#6b7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                Segmentation
              </div>
              <button type="button" role="menuitem" disabled className="btn btn--ghost btn--sm">
                Binarize
              </button>
              <label
                role="menuitem"
                className="btn btn--ghost btn--sm"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}
              >
                <input
                  type="checkbox"
                  aria-label="Refine Kraken segments with Otsu"
                  checked={useOtsuRefinement}
                  disabled={segmenting}
                  onChange={(event) => onUseOtsuRefinementChange(event.target.checked)}
                />
                Otsu refine
              </label>
              <button
                type="button"
                role="menuitem"
                disabled={segmenting || ocrRunning}
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunAutoSegment();
                }}
                className="btn btn--ghost btn--sm"
              >
                {segmenting ? 'Segmenting…' : 'Auto segment'}
              </button>
              <div style={{ height: 1, background: '#e5e7eb', marginBlock: 2 }} />
              <div
                style={{
                  padding: '2px 6px',
                  fontSize: 11,
                  fontWeight: 600,
                  color: '#6b7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                OCR
              </div>
              {transcribeModels.length > 0 && (
                <label
                  className="btn btn--ghost btn--sm"
                  style={{ display: 'inline-flex', justifyContent: 'space-between', gap: 8 }}
                >
                  <span>OCR model</span>
                  <select
                    aria-label="OCR model"
                    value={selectedTranscribeModelId ?? ''}
                    disabled={ocrRunning}
                    onChange={(event) =>
                      onSelectedTranscribeModelIdChange(event.target.value || null)
                    }
                    style={{ minWidth: 0, flex: 1 }}
                  >
                    {transcribeModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <button
                type="button"
                role="menuitem"
                disabled={!selectedSegmentId || ocrRunning || segmenting || !selectedTranscribeModelId}
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunSegmentOcr();
                }}
                className="btn btn--ghost btn--sm"
              >
                {ocrRunning ? 'OCR…' : 'OCR segment'}
              </button>
              <button
                type="button"
                role="menuitem"
                disabled={ocrRunning || segmenting || lines.length === 0 || !selectedTranscribeModelId}
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunPageOcr();
                }}
                className="btn btn--ghost btn--sm"
              >
                {ocrRunning ? 'OCR…' : 'OCR page'}
              </button>
            </div>
          )}
        </div>
        <div style={toolbarClusterStyle}>
          <button
            type="button"
            onClick={() => void onUpdateReviewStatus(!part.reviewed)}
            className="btn btn--ghost btn--sm"
          >
            {part.reviewed ? 'Mark unreviewed' : 'Mark reviewed'}
          </button>
          <button type="button" disabled className="btn btn--primary btn--sm">
            Export
          </button>
        </div>
      </div>
    </header>
  );
}
