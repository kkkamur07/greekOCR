/**
 * Hand a blob the API already returned to the browser as a file.
 *
 * The export routes are Bearer-authenticated, so a plain `<a href>` or a
 * `window.open` cannot reach them: a browser-initiated navigation carries no
 * Authorization header and the request comes back 401. The bytes therefore
 * have to travel through the API client first (which attaches the token and
 * can refresh it), and only then become a download. This is the same shape
 * `PageEditorPageXmlButton` uses for the single-page export, lifted so the
 * document-level exports do not each grow their own copy.
 *
 * The object URL is revoked straight after the synthetic click. The click
 * hands the blob to the browser's download machinery synchronously, so
 * revoking on the next line does not cut the save short.
 */
export function saveBlobAsFile(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  try {
    const anchor = globalThis.document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}
