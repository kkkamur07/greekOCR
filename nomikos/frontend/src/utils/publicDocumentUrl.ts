// The share token is part of the link itself, not a header a browser tab
// carries around, so a caller with no token has no working link to build -
// see DocumentLiveSharingControls for the "owner only" message that stands
// in for it instead.
export function publicDocumentPath(
  projectId: string,
  documentId: string,
  token: string,
): string {
  return `/public/projects/${projectId}/documents/${documentId}?t=${encodeURIComponent(token)}`;
}

export function publicDocumentUrl(
  projectId: string,
  documentId: string,
  token: string,
): string {
  return `${window.location.origin}${publicDocumentPath(projectId, documentId, token)}`;
}
