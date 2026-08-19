/**
 * Client-side PDF splitting: one dropped PDF becomes one PNG per page.
 *
 * The split happens in the browser for the same reason `encodePartImage`
 * exists: the production API is serverless and caps a request body at 4.5 MB,
 * so a scanned PDF could never reach a server-side splitter. Each rendered
 * page instead travels the existing part-upload path, presigned direct-to-
 * storage included.
 *
 * Pages render at 300 DPI - the resolution OCR pipelines expect of a scan -
 * and export as PNG, because the stored image is ground truth and this
 * module must not add a second lossy generation on top of whatever the PDF
 * already contains (see `encodePartImage` for the same rule).
 */

import { pdfPageFileName } from "./uploadBatch";

/** PDF user space is defined as 72 units per inch. */
const PDF_UNITS_PER_INCH = 72;
const TARGET_RENDER_DPI = 300;

/**
 * Upper bound on rendered pixels per page. 40 MP holds an A2 sheet at
 * 300 DPI, keeps the PNG comfortably under the server's 100 MB part cap
 * (`MAX_PART_UPLOAD_BYTES`), and stays inside every desktop browser's canvas
 * allocation limit. An outsized page renders at whatever DPI fits.
 */
export const MAX_PDF_RENDER_PIXELS = 40_000_000;

export class PdfReadError extends Error {
  constructor(message = "Could not read the PDF") {
    super(message);
    this.name = "PdfReadError";
  }
}

/**
 * pdf.js parses in a Web Worker so a large PDF does not freeze the tab. The
 * worker is created once and only where workers exist at all - under jsdom
 * (tests) pdf.js falls back to parsing on the main thread.
 */
async function loadPdfjs() {
  const pdfjs = await import("pdfjs-dist");
  if (typeof Worker !== "undefined" && !pdfjs.GlobalWorkerOptions.workerPort) {
    pdfjs.GlobalWorkerOptions.workerPort = new Worker(
      new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url),
      { type: "module" },
    );
  }
  return pdfjs;
}

/** The scale that renders a page at 300 DPI, lowered until it fits the pixel cap. */
export function renderScaleFor(
  widthUnits: number,
  heightUnits: number,
): number {
  const target = TARGET_RENDER_DPI / PDF_UNITS_PER_INCH;
  const capped = Math.sqrt(MAX_PDF_RENDER_PIXELS / (widthUnits * heightUnits));
  return Math.min(target, capped);
}

/**
 * Render every page of `file` to a PNG `File`, in document order.
 *
 * `onProgress(done, total)` fires after each page so the caller can narrate
 * a long split. Rejects with `PdfReadError` when the file is not a PDF the
 * browser can parse; a failure on one page fails the whole file rather than
 * silently uploading a document with holes in it.
 */
export async function renderPdfToPageFiles(
  file: File,
  onProgress?: (done: number, total: number) => void,
): Promise<File[]> {
  const pdfjs = await loadPdfjs();
  const data = await file.arrayBuffer();

  const loadingTask = pdfjs.getDocument({ data });
  let doc;
  try {
    doc = await loadingTask.promise;
  } catch {
    throw new PdfReadError(`Could not read ${file.name} as a PDF`);
  }

  try {
    const pages: File[] = [];
    for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber += 1) {
      const page = await doc.getPage(pageNumber);
      try {
        const base = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({
          scale: renderScaleFor(base.width, base.height),
        });

        const canvas = document.createElement("canvas");
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        const canvasContext = canvas.getContext("2d");
        if (!canvasContext) {
          throw new PdfReadError();
        }

        await page.render({ canvas, canvasContext, viewport }).promise;

        const blob = await new Promise<Blob>((resolve, reject) => {
          canvas.toBlob((result) => {
            if (result) resolve(result);
            else reject(new PdfReadError(`Could not render ${file.name}`));
          }, "image/png");
        });
        // Release the bitmap now rather than when GC notices; a hundred-page
        // split would otherwise hold a hundred full-page canvases.
        canvas.width = 0;
        canvas.height = 0;

        pages.push(
          new File(
            [blob],
            pdfPageFileName(file.name, pageNumber, doc.numPages),
            {
              type: "image/png",
            },
          ),
        );
      } finally {
        page.cleanup();
      }
      onProgress?.(pageNumber, doc.numPages);
    }
    return pages;
  } finally {
    // Frees the parse worker's memory for this document; the worker itself
    // is shared and stays alive for the next drop.
    void loadingTask.destroy();
  }
}
