/**
 * The file stem every export of one page shares: `<Document_name>_page_<n>`.
 *
 * Mirrors `export_file_stem` in the backend's PAGE XML export service, so the
 * archive the browser saves, the XML and the image inside it, and the name the
 * server would have suggested all agree. Characters no common filesystem
 * accepts are dropped, runs of whitespace collapse to one underscore, and the
 * document part is capped so a long title cannot push the name past path
 * limits. Non-ASCII letters stay: a Greek title is a Greek title.
 */
const UNSAFE_FILENAME_CHARS = /[\\/:*?"<>|\p{Cc}]+/gu;
const MAX_STEM_DOCUMENT_CHARS = 80;

export function exportFileStem(
  documentName: string,
  pageNumber: number,
): string {
  const safe = documentName
    .replace(UNSAFE_FILENAME_CHARS, "")
    .trim()
    .replace(/\s+/g, "_")
    .slice(0, MAX_STEM_DOCUMENT_CHARS)
    .replace(/[._]+$/, "");
  return `${safe || "document"}_page_${pageNumber}`;
}
