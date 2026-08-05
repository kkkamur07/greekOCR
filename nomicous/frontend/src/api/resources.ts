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
import { invalidateTags, type ResourceTag } from "./resourceCache";

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
 */
export const invalidateAfter = {
  projectCreated: (): void => invalidateTags([resourceTags.projects]),

  projectUpdated: (projectId: string): void =>
    invalidateTags([resourceTags.projects, resourceTags.project(projectId)]),

  /**
   * Same write, by a caller that already folded the server's response into its
   * own view with `ServerQuery.patch`. Dropping that view would replace the
   * value just written with an older one, so only the reads that copy the
   * project's fields elsewhere - the project list - are invalidated.
   */
  projectUpdatedInPlace: (): void => invalidateTags([resourceTags.projects]),

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

  /** As `documentUpdated`, for a caller that holds the response - see `projectUpdatedInPlace`. */
  documentUpdatedInPlace: (
    projectId: string,
    documentId: string,
  ): void =>
    invalidateTags([
      resourceTags.documents(projectId),
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
} as const;
