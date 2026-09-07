type PageEditorPagerProps = {
  /** 1-based, or null while the document has not resolved the page yet. */
  pageNumber: number | null;
  pageCount: number;
  hasPreviousPart: boolean;
  hasNextPart: boolean;
  onPreviousPart: () => void;
  onNextPart: () => void;
};

/**
 * Move a page at a time without leaving the editor.
 *
 * Plain buttons, so Enter and Space reach them for free, and the shortcut is
 * named in the title where a researcher will look for it rather than only in a
 * help panel nobody opens.
 */
export function PageEditorPager({
  pageNumber,
  pageCount,
  hasPreviousPart,
  hasNextPart,
  onPreviousPart,
  onNextPart,
}: PageEditorPagerProps) {
  return (
    <div className="pe-pager" role="group" aria-label="Page navigation">
      <button
        type="button"
        className="pe-pager__step"
        onClick={onPreviousPart}
        disabled={!hasPreviousPart}
        aria-label="Previous page"
        title="Previous page (Page Up)"
      >
        ‹
      </button>
      <span className="pe-pager__label">
        p.{pageNumber ?? "-"}
        <span className="pe-pager__of"> / {pageCount}</span>
      </span>
      <button
        type="button"
        className="pe-pager__step"
        onClick={onNextPart}
        disabled={!hasNextPart}
        aria-label="Next page"
        title="Next page (Page Down)"
      >
        ›
      </button>
    </div>
  );
}
