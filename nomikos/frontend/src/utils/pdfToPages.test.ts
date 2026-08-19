/**
 * The pdf.js interaction is faked; what these tests pin down is ours: page
 * files come back in document order under sortable names, the render scale
 * targets 300 DPI but yields to the pixel cap, and an unparseable file
 * surfaces as `PdfReadError` rather than a stuck upload.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  MAX_PDF_RENDER_PIXELS,
  PdfReadError,
  renderPdfToPageFiles,
  renderScaleFor,
} from "./pdfToPages";

const renderedScales: number[] = [];
let failParse = false;

vi.mock("pdfjs-dist", () => {
  const page = {
    getViewport: ({ scale }: { scale: number }) => ({
      width: 612 * scale,
      height: 792 * scale,
      scale,
    }),
    render: ({ viewport }: { viewport: { scale: number } }) => {
      renderedScales.push(viewport.scale);
      return { promise: Promise.resolve() };
    },
    cleanup: () => {},
  };
  return {
    GlobalWorkerOptions: { workerPort: null },
    getDocument: () => ({
      promise: failParse
        ? Promise.reject(new Error("bad pdf"))
        : Promise.resolve({
            numPages: 3,
            getPage: () => Promise.resolve(page),
          }),
      destroy: () => Promise.resolve(),
    }),
  };
});

beforeEach(() => {
  renderedScales.length = 0;
  failParse = false;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    {} as CanvasRenderingContext2D,
  );
  // jsdom has no toBlob; the real one hands back the canvas's PNG bytes.
  Object.defineProperty(HTMLCanvasElement.prototype, "toBlob", {
    configurable: true,
    value: (cb: (blob: Blob | null) => void) =>
      cb(new Blob(["png-bytes"], { type: "image/png" })),
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("renderScaleFor", () => {
  it("targets 300 DPI for an ordinary page", () => {
    // US Letter: 612x792 PDF units.
    expect(renderScaleFor(612, 792)).toBeCloseTo(300 / 72, 5);
  });

  it("yields to the pixel cap for an outsized page", () => {
    const scale = renderScaleFor(20_000, 20_000);
    expect(scale).toBeLessThan(300 / 72);
    expect(20_000 * scale * (20_000 * scale)).toBeLessThanOrEqual(
      MAX_PDF_RENDER_PIXELS * 1.001,
    );
  });
});

describe("renderPdfToPageFiles", () => {
  it("renders every page to an ordered, sortably named PNG file", async () => {
    const progress: Array<[number, number]> = [];
    const files = await renderPdfToPageFiles(
      new File(["%PDF"], "scan.pdf", { type: "application/pdf" }),
      (done, total) => progress.push([done, total]),
    );

    expect(files.map((f) => f.name)).toEqual([
      "scan-p1.png",
      "scan-p2.png",
      "scan-p3.png",
    ]);
    expect(files.every((f) => f.type === "image/png")).toBe(true);
    expect(progress).toEqual([
      [1, 3],
      [2, 3],
      [3, 3],
    ]);
    expect(renderedScales).toEqual([300 / 72, 300 / 72, 300 / 72]);
  });

  it("rejects with PdfReadError when the file cannot be parsed", async () => {
    failParse = true;
    await expect(
      renderPdfToPageFiles(new File(["junk"], "junk.pdf")),
    ).rejects.toBeInstanceOf(PdfReadError);
  });
});
