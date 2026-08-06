import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { PublicCanvasPdfView } from "./PublicCanvasPdfView";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getPublicTranscriptionPdf: vi.fn(),
    },
  };
});

vi.mock("./PublicZoomSurface", () => ({
  PublicZoomSurface: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

const mockedGetPublicTranscriptionPdf =
  api.getPublicTranscriptionPdf as ReturnType<typeof vi.fn>;

describe("PublicCanvasPdfView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetPublicTranscriptionPdf.mockResolvedValue(
      new Blob(["%PDF"], { type: "application/pdf" }),
    );
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:public-preview"),
      revokeObjectURL: vi.fn(),
    });
  });

  it("embeds with an iframe, never an object that object-src 'none' blocks", async () => {
    const { container } = render(
      <PublicCanvasPdfView
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
      />,
    );

    const embed = await screen.findByTitle("Transcription PDF");
    expect(embed.tagName).toBe("IFRAME");
    expect(embed).toHaveAttribute("src", "blob:public-preview");
    expect(container.querySelector("object")).toBeNull();
    expect(container.querySelector("embed")).toBeNull();

    vi.unstubAllGlobals();
  });

  it("offers open-in-new-tab and download outside the embed", async () => {
    render(
      <PublicCanvasPdfView
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
      />,
    );

    // The embed cannot report its own failure - <iframe> fallback content does
    // not fire when a blob frame is refused - so these must not live inside it.
    const openLink = await screen.findByRole("link", {
      name: /open pdf in new tab/i,
    });
    expect(openLink).toHaveAttribute("href", "blob:public-preview");
    expect(openLink).toHaveAttribute("target", "_blank");
    expect(openLink).toHaveAttribute(
      "rel",
      expect.stringContaining("noopener"),
    );
    expect(openLink.closest("iframe")).toBeNull();

    const downloadLink = screen.getByRole("link", { name: /download pdf/i });
    expect(downloadLink).toHaveAttribute("href", "blob:public-preview");
    expect(downloadLink).toHaveAttribute("download", "transcription.pdf");
    expect(downloadLink.closest("iframe")).toBeNull();

    vi.unstubAllGlobals();
  });

  it("uses the caller's download filename when one is given", async () => {
    render(
      <PublicCanvasPdfView
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="page-4.pdf"
      />,
    );

    expect(
      await screen.findByRole("link", { name: /download pdf/i }),
    ).toHaveAttribute("download", "page-4.pdf");

    vi.unstubAllGlobals();
  });

  it("surfaces API errors instead of an empty frame", async () => {
    mockedGetPublicTranscriptionPdf.mockRejectedValue(
      new ApiError("Not found", 404),
    );

    const { container } = render(
      <PublicCanvasPdfView
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
      />,
    );

    expect(await screen.findByText("Not found")).toBeTruthy();
    expect(container.querySelector("iframe")).toBeNull();

    vi.unstubAllGlobals();
  });
});

/**
 * The components above are only half the fix. A `blob:` embed is blocked at the
 * header, so the shipped policy is part of the contract and is asserted here
 * rather than left to a deploy to discover.
 */
describe("shipped frontend CSP", () => {
  // Not `new URL(relative, import.meta.url)`: under the jsdom environment the
  // global URL resolves relatives against the document base, not the module.
  const vercelJson = readFileSync(
    join(
      dirname(fileURLToPath(import.meta.url)),
      "..",
      "..",
      "..",
      "vercel.json",
    ),
    "utf-8",
  ) as string;

  function contentSecurityPolicy(): Map<string, string[]> {
    const config = JSON.parse(vercelJson) as {
      headers: { headers: { key: string; value: string }[] }[];
    };
    const header = config.headers
      .flatMap((entry) => entry.headers)
      .find((entry) => entry.key === "Content-Security-Policy");
    expect(header).toBeDefined();

    const directives = new Map<string, string[]>();
    for (const directive of header!.value.split(";")) {
      const [name, ...values] = directive.trim().split(/\s+/);
      if (name) directives.set(name, values);
    }
    return directives;
  }

  it("permits same-origin blob frames so the PDF preview can render", () => {
    const frameSrc = contentSecurityPolicy().get("frame-src");

    // `blob:` is a scheme of its own: it is not covered by `'self'`, and not by
    // the `default-src 'self'` fallback `frame-src` would otherwise inherit.
    // That is why img-src and worker-src already name it explicitly.
    expect(frameSrc).toBeDefined();
    expect(frameSrc).toContain("blob:");
    expect(frameSrc).toContain("'self'");
  });

  it("grants no remote origin the right to be framed", () => {
    for (const value of contentSecurityPolicy().get("frame-src") ?? []) {
      expect(value.startsWith("http")).toBe(false);
    }
  });

  it("keeps object-src 'none' - the embed moved, the directive did not relax", () => {
    // Re-permitting plugin content would have been the larger concession.
    expect(contentSecurityPolicy().get("object-src")).toEqual(["'none'"]);
  });

  it("still refuses to be framed by anyone", () => {
    expect(contentSecurityPolicy().get("frame-ancestors")).toEqual(["'none'"]);
  });
});
