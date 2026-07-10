import type { DocumentWithPartsResponse } from "../api/client";

export type PageEditorLocationState = {
  document?: DocumentWithPartsResponse;
};

export function readPageEditorDocument(
  state: unknown,
  projectId: string,
  documentId: string,
): DocumentWithPartsResponse | null {
  if (!state || typeof state !== "object") return null;
  const document = (state as PageEditorLocationState).document;
  if (
    !document ||
    document.id !== documentId ||
    document.project_id !== projectId
  ) {
    return null;
  }
  return document;
}
