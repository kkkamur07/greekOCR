import type { PublicLayoutResponse } from '../../api/client';
import { lineTextForLayer, linesForPart } from '../../utils/publicLayout';

type PublicTranscriptPanelProps = {
  partId: string;
  layout: PublicLayoutResponse | null;
  selectedLineIndex: number | null;
  onSelectLine: (index: number | null) => void;
};

export function PublicTranscriptPanel({
  partId,
  layout,
  selectedLineIndex,
  onSelectLine,
}: PublicTranscriptPanelProps) {
  const partLines = linesForPart(layout?.lines, partId);
  const selectedLine =
    selectedLineIndex !== null && selectedLineIndex >= 0
      ? (partLines[selectedLineIndex] ?? null)
      : null;

  return (
    <aside className="pub-transcript" aria-labelledby="transcript-heading">
      <h2 className="pub-transcript__heading" id="transcript-heading">
        {selectedLine ? `Line ${selectedLine.order + 1}` : 'Transcription'}
      </h2>

      {partLines.length === 0 ? (
        <p className="text-sm text-muted">No line geometry on this page yet.</p>
      ) : (
        <div className="pub-line-list" role="list" aria-label="Page transcription lines">
          {partLines.map((line, index) => {
            const text = lineTextForLayer(line, null);
            const selected = selectedLineIndex === index;
            return (
              <button
                key={line.id}
                type="button"
                role="listitem"
                className={`text-block text-block--interactive${selected ? ' text-block--selected' : ''}`}
                aria-pressed={selected}
                onClick={() => onSelectLine(selected ? null : index)}
              >
                {text ?? (
                  <span className="text-block__empty">No transcription for this line</span>
                )}
              </button>
            );
          })}
        </div>
      )}

      <p className="pub-transcript__hint">
        {partLines.length} line{partLines.length === 1 ? '' : 's'} on this page
      </p>
    </aside>
  );
}
