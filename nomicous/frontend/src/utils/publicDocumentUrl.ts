export function publicDocumentPath(projectId: string, documentId: string): string {
  return `/public/projects/${projectId}/documents/${documentId}`;
}

export function publicDocumentUrl(projectId: string, documentId: string): string {
  return `${window.location.origin}${publicDocumentPath(projectId, documentId)}`;
}
