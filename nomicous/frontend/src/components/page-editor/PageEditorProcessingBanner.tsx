export type PageEditorProcessingKind =
  | 'segmentation'
  | 'transcription-page'
  | 'transcription-segment'
  | null;

export function getPageEditorProcessingLabel(kind: PageEditorProcessingKind): string | null {
  switch (kind) {
    case 'segmentation':
      return 'Segmentation in progress';
    case 'transcription-page':
      return 'Full-page transcription in progress';
    case 'transcription-segment':
      return 'Segment transcription in progress';
    default:
      return null;
  }
}

type PageEditorProcessingBannerProps = {
  kind: PageEditorProcessingKind;
};

export function PageEditorProcessingBanner({ kind }: PageEditorProcessingBannerProps) {
  const label = getPageEditorProcessingLabel(kind);
  if (!label) return null;

  return (
    <div className="pe-processing-banner" role="status" aria-live="polite">
      <span className="pe-processing-banner__spinner" aria-hidden="true" />
      <span className="pe-processing-banner__label">{label}</span>
      <span className="pe-processing-banner__hint">
        Track progress in the jobs panel (bottom-right).
      </span>
    </div>
  );
}
