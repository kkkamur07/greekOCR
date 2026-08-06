/**
 * The resource vocabulary of the server-state layer.
 *
 * `resourceTags` names each server resource once, and every cached read declares
 * which of them it depends on. `invalidateAfter` then declares, once per write,
 * the tags that write dirties.
 *
 * The point of the split is that a call site names *the write it performed* -
 * `invalidateAfter.documentDeleted(projectId, documentId)` - and never
 * enumerates the reads that happen to depend on it. Adding a new read is
 * therefore not a change every existing mutation has to remember; it only has to
 * declare its tags. The old failure mode, where a mutation quietly forgot to
 * refresh a list and the UI showed stale data, has nowhere left to live.
 */
import {
  invalidateResourceTags as invalidateTags,
  type ResourceTag,
} from "./queryClient";

export type { ResourceTag };

export const resourceTags = {
  currentUser: "current-user",
  projects: "projects",
  project: (projectId: string): ResourceTag => `project:${projectId}`,
  documents: (projectId: string): ResourceTag => `documents:${projectId}`,
  document: (projectId: string, documentId: string): ResourceTag =>
    `document:${projectId}:${documentId}`,
  publicDocument: (projectId: string, documentId: string): ResourceTag =>
    `public-document:${projectId}:${documentId}`,
} as const;

/**
 * A project carries its own `document_count`, so anything that adds or removes a
 * document makes the project list stale too. That relationship is written down
 * here once rather than being rediscovered at each call site that creates or
 * deletes a document.
 *
 * A caller that already folded the server's response into its own view with
 * `ServerQuery.patch` still declares the whole write. There used to be a
 * narrower `…InPlace` pair for that case, on the reasoning that invalidating a
 * view you just wrote to replaces it with an older value - it does not, because
 * the refetch reads the same server the response came from. What the narrower
 * pair did do was skip a tag, and the reads carrying that tag - the other
 * `includeArchived` variant of the dashboard, the copy of the document the page
 * editor holds - were left showing the value from before the write.
 */
export const invalidateAfter = {
  projectCreated: (): void => invalidateTags([resourceTags.projects]),

  projectUpdated: (projectId: string): void =>
    invalidateTags([resourceTags.projects, resourceTags.project(projectId)]),

  projectDeleted: (projectId: string): void =>
    invalidateTags([resourceTags.projects, resourceTags.project(projectId)]),

  documentCreated: (projectId: string): void =>
    invalidateTags([resourceTags.projects, resourceTags.documents(projectId)]),

  documentUpdated: (projectId: string, documentId: string): void =>
    invalidateTags([
      resourceTags.documents(projectId),
      resourceTags.document(projectId, documentId),
      resourceTags.publicDocument(projectId, documentId),
    ]),

  documentDeleted: (projectId: string, documentId: string): void =>
    invalidateTags([
      resourceTags.projects,
      resourceTags.documents(projectId),
      resourceTags.document(projectId, documentId),
      resourceTags.publicDocument(projectId, documentId),
    ]),

  /** Upload, delete, reorder, review status: the document's parts changed. */
  documentPartsChanged: (projectId: string, documentId: string): void =>
    invalidateTags([
      resourceTags.documents(projectId),
      resourceTags.document(projectId, documentId),
      resourceTags.publicDocument(projectId, documentId),
    ]),

  /**
   * Segments, layout or transcriptions on one page changed.
   *
   * The page editor holds its own copies of those and writes them straight
   * back, so it is easy to forget that two cached reads are copies of the same
   * thing: the document the editor and the detail page share, and the published
   * page a reader sees. Neither is in the editor's own state, and neither
   * refreshes on its own inside the freshness window.
   *
   * The project's document list is not touched: nothing in it changes when a
   * line is redrawn.
   */
  partContentChanged: (projectId: string, documentId: string): void =>
    invalidateTags([
      resourceTags.document(projectId, documentId),
      resourceTags.publicDocument(projectId, documentId),
    ]),
} as const;
