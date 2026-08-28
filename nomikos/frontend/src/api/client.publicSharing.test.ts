/**
 * Every `/public/*` route 404s without `?t=<public_share_token>` (see the
 * backend's `access.py`), so each public getter has to carry the token in its
 * query string rather than assume the caller's cookie or header does it.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, publicPartMediaUrl } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function blobResponse(status = 200): Response {
  return new Response("bytes", { status });
}

/** The path (with query string) the mocked fetch was actually asked for. */
function requestedPath(fetchMock: { mock: { calls: unknown[][] } }): string {
  const [url] = fetchMock.mock.calls[0] as [string];
  return new URL(url).pathname + new URL(url).search;
}

describe("public getters carry the share token", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("appends t to the document request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicDocument("project-1", "doc-1", "share-token-1");

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1?t=share-token-1",
    );
  });

  it("appends t to the layout request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicLayout("project-1", "doc-1", "share-token-1");

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1/layout?t=share-token-1",
    );
  });

  it("appends t to the transcriptions request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await api.listPublicTranscriptions("project-1", "doc-1", "share-token-1");

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1/transcriptions?t=share-token-1",
    );
  });

  it("appends t to the transcription PDF request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(blobResponse());
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicTranscriptionPdf(
      "project-1",
      "doc-1",
      "part-1",
      "share-token-1",
    );

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1/parts/part-1/transcription-pdf?t=share-token-1",
    );
  });

  it("appends t to the PAGE XML request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(blobResponse());
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicPageXml("project-1", "doc-1", "part-1", "share-token-1");

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1/parts/part-1/page-xml?t=share-token-1",
    );
  });

  it("appends t to the PAGE XML bundle request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(blobResponse());
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicPageXmlBundle(
      "project-1",
      "doc-1",
      "part-1",
      "share-token-1",
    );

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1/parts/part-1/page-xml-bundle?t=share-token-1",
    );
  });

  it("leaves the request bare when there is no token to send", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicDocument("project-1", "doc-1", null);

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1",
    );
  });

  it("adds t to the part image URL", () => {
    expect(publicPartMediaUrl("part-1", "share-token-1")).toMatch(
      /\/public\/media\/parts\/part-1\?t=share-token-1$/,
    );
  });

  it("leaves the part image URL bare with no token", () => {
    expect(publicPartMediaUrl("part-1", null)).toMatch(
      /\/public\/media\/parts\/part-1$/,
    );
  });

  it("url-encodes a token that needs it", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await api.getPublicDocument("project-1", "doc-1", "a token/with+chars");

    expect(requestedPath(fetchMock)).toBe(
      "/public/projects/project-1/documents/doc-1?t=a%20token%2Fwith%2Bchars",
    );
  });
});
