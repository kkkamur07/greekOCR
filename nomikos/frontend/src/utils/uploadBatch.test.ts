import { describe, expect, it } from "vitest";

import { compareFilenames, isPdfFile, pdfPageFileName } from "./uploadBatch";

describe("compareFilenames", () => {
  it("orders numeric runs as numbers, not text", () => {
    const names = ["page-10.jpg", "page-2.jpg", "page-1.jpg"];
    names.sort(compareFilenames);
    expect(names).toEqual(["page-1.jpg", "page-2.jpg", "page-10.jpg"]);
  });

  it("ignores case, as file managers do", () => {
    expect(compareFilenames("Page-1.jpg", "page-2.jpg")).toBeLessThan(0);
  });
});

describe("isPdfFile", () => {
  it("accepts the PDF MIME type", () => {
    expect(isPdfFile(new File([], "scan", { type: "application/pdf" }))).toBe(
      true,
    );
  });

  it("falls back to the extension when the type is empty", () => {
    // Files dragged from some file managers arrive with no MIME type at all.
    expect(isPdfFile(new File([], "scan.PDF", { type: "" }))).toBe(true);
  });

  it("rejects images", () => {
    expect(isPdfFile(new File([], "scan.jpg", { type: "image/jpeg" }))).toBe(
      false,
    );
  });
});

describe("pdfPageFileName", () => {
  it("zero-pads to the page count's width so filename order is page order", () => {
    expect(pdfPageFileName("scan.pdf", 3, 120)).toBe("scan-p003.png");
    expect(pdfPageFileName("scan.pdf", 40, 120)).toBe("scan-p040.png");
  });

  it("drops the .pdf extension case-insensitively", () => {
    expect(pdfPageFileName("Grec1360.PDF", 1, 3)).toBe("Grec1360-p1.png");
  });
});
