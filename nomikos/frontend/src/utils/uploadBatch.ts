/**
 * Ordering and naming rules for multi-file part uploads.
 *
 * A drop can carry any mix of page images and PDFs. The document's page order
 * comes from filenames, not from the order the browser happens to enumerate
 * the drop in - researchers name their scans `page-1`, `page-2`, ..., and a
 * scan set sorted as text would interleave `page-10` between `page-1` and
 * `page-2`. PDF pages have no filenames of their own, so they inherit the
 * PDF's position in that ordering and keep their in-document sequence.
 */

/** Filename order with numeric runs compared as numbers: page-2 before page-10. */
export function compareFilenames(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

/**
 * Whether a dropped file is a PDF. Type sniffing alone is not enough: a file
 * dragged out of some file managers arrives with an empty MIME type, leaving
 * the extension as the only signal.
 */
export function isPdfFile(file: File): boolean {
  return file.type === "application/pdf" || /\.pdf$/i.test(file.name);
}

/**
 * The filename a rendered PDF page uploads under, e.g. `scan-p03.png`.
 *
 * Page numbers are zero-padded to the page count's width so the generated
 * names keep their document order under the same filename sort that orders
 * the rest of the drop.
 */
export function pdfPageFileName(
  pdfName: string,
  pageNumber: number,
  pageCount: number,
): string {
  const stem = pdfName.replace(/\.pdf$/i, "");
  const width = String(pageCount).length;
  return `${stem || pdfName}-p${String(pageNumber).padStart(width, "0")}.png`;
}
