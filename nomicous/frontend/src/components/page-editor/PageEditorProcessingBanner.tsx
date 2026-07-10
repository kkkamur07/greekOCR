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
