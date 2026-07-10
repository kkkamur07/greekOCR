type PageEditorSegmentNavProps = {
  segmentNumber: number | null;
  totalSegments: number;
  onPrevious: () => void;
  onNext: () => void;
  disabled?: boolean;
};

export function PageEditorSegmentNav({
  segmentNumber,
  totalSegments,
  onPrevious,
  onNext,
  disabled = false,
}: PageEditorSegmentNavProps) {
  const atStart = segmentNumber === null || segmentNumber <= 1;
  const atEnd = segmentNumber === null || segmentNumber >= totalSegments;
  const label =
    segmentNumber !== null && totalSegments > 0
      ? `${segmentNumber} / ${totalSegments}`
      : totalSegments > 0
        ? `- / ${totalSegments}`
        : "-";

  return (
    <nav className="pe-seg-nav" aria-label="Segment navigation">
      <button
        type="button"
        className="pe-seg-nav__btn"
        aria-label="Previous segment"
        disabled={disabled || atStart || totalSegments === 0}
        onClick={onPrevious}
      >
        ‹
      </button>
      <span className="pe-seg-nav__label" aria-live="polite">
        {label}
      </span>
      <button
        type="button"
        className="pe-seg-nav__btn"
        aria-label="Next segment"
        disabled={disabled || atEnd || totalSegments === 0}
        onClick={onNext}
      >
        ›
      </button>
    </nav>
  );
}
