# Annote Production Merge — Product Requirements Document

## Problem Statement

Researchers need annote's focused manuscript Page editor, Segment drawing, Pairing workflow, Annotation history, and Export behavior inside the authenticated Project and Document hierarchy that already exists in the platform. Today those capabilities are split: annote is a standalone filesystem-backed app, while the platform has users, auth, projects, documents, Postgres, OpenAPI, and a frontend shell.

This split makes it hard to collaborate, preserve work in the database, review Pages consistently, build training-ready line data from approved Ground truth transcription, and implement production deployment cleanly. The final production app must live under annote while keeping the root model workspace separate.

## Solution

Merge the platform backend, frontend, and infrastructure into annote as the production app. The merged backend follows DDD bounded contexts: users, project, document, annotation, and inference. The document context owns the canonical manuscript hierarchy: Project -> Document -> Document part/Page -> Line/Segment -> Line transcription. The annotation context owns editor workflows such as Pairing progress, Review status, compact Annotation history, and Export.

The merged frontend uses the platform app shell and API-driven routing while preserving annote's editor theme and workflow. Researchers can log in, open a Project, open a Document, edit a Document part/Page, draw and edit Lines/Segments, pair approved text per Line, track Pairing progress, mark the Page reviewed or unreviewed, restore compact Annotation history, and export Processed line images with Line transcription files. OCR prediction execution is deferred to a focused design pass.

## User Stories

1. As a researcher, I want to register and log in, so that my annotation work is private by default.
2. As a researcher, I want to create a Project, so that I can group related manuscript work.
3. As a project owner, I want to share a Project with collaborators, so that multiple humans can work on the same corpus.
4. As a collaborator, I want to see only Projects I own or that were shared with me, so that other work remains protected.
5. As a researcher, I want to create a Document inside a Project, so that one manuscript or work has a durable home.
6. As a researcher, I want a Document to contain multiple Document parts, so that each Page image can be edited in order.
7. As a researcher, I want one Document part to represent one Page image, so that the hierarchy matches manuscript work.
8. As a researcher, I want to upload Page images into a Document, so that I can start annotating without touching local data folders.
9. As a researcher, I want existing root-level model research code to stay separate, so that production app work does not disturb training experiments.
10. As an operator, I want production backend, frontend, and infrastructure to live under annote, so that deployment has one app root.
11. As a developer, I want duplicate root app folders removed after merge, so that there is only one production app to maintain.
12. As a developer, I want infrastructure separated from backend code, so that database migrations and future Terraform can live together operationally.
13. As a researcher, I want to open a Document part/Page in the annote editor theme, so that the familiar annotation workflow is preserved.
14. As a researcher, I want to draw a Polygon segment around one written line of ink, so that the platform can persist it as a Line.
15. As a researcher, I want to draw a Rectangle segment when appropriate, so that simple line regions are quick to create.
16. As a researcher, I want Segment geometry to persist as Line geometry, so that no separate Segment table splits the domain model.
17. As a researcher, I want to edit Segment vertices, so that I can correct geometry after drawing or model creation.
18. As a researcher, I want Segment numbers to remain stable enough for exports, so that generated line artifacts are understandable.
19. As a researcher, I want to delete incorrect Lines/Segments, so that mistakes do not remain in the Page.
20. As a researcher, I want a Page transcription import helper, so that pasted or uploaded text can be split into candidate Text lines.
21. As a researcher, I want Page transcription text to be partial, so that I can work on incomplete Pages.
22. As a researcher, I want candidate Text lines to remain separate from canonical Ground truth transcription, so that imports do not become truth automatically.
23. As a researcher, I want to select a Segment first and then assign a Text line, so that Pairing follows the annote workflow.
24. As a researcher, I want to type text directly for a selected Segment, so that I can create Ground truth transcription without an import file.
25. As a researcher, I want approved text stored per Line, so that partially completed Pages are represented accurately.
26. As a researcher, I want Transcription to be document-level, so that Ground truth can span all Document parts.
27. As a researcher, I want Line transcription to link one Line to one document-level Transcription, so that per-line text is canonical.
28. As a researcher, I want Model transcription kept separate from Ground truth transcription, so that OCR output does not overwrite human work.
29. As a researcher, I want to accept or edit a Model transcription into Ground truth later, so that OCR can assist without becoming authoritative.
30. As a researcher, I want Pairing progress visible on a Page, so that I know how much of the Page has approved text.
31. As a researcher, I want Pairing progress independent from Human review, so that coverage and trust are separate signals.
32. As a reviewer, I want to mark a Page reviewed or unreviewed, so that I can record whether current Ground truth transcriptions have been checked.
33. As a reviewer, I want Review status stored as a boolean, so that the backend remains simple.
34. As a reviewer, I want the frontend to show Reviewed and Unreviewed labels, so that the boolean is readable.
35. As a reviewer, I want a partially paired Page to be markable as reviewed, so that reviewed partial work is supported.
36. As a researcher, I want edits to remain human-controlled without an automatic lock workflow, so that I can keep correcting Pages freely.
37. As a researcher, I want Annotation history, so that I can recover from accidental edits or overwrites.
38. As a researcher, I want Annotation history stored compactly, so that restore is useful without storing images or generated exports.
39. As a researcher, I want to restore a History snapshot, so that a prior Page annotation state can replace the current one.
40. As a developer, I want Annotation history in the annotation context, so that editor recovery does not pollute the canonical document model.
41. As a researcher, I want Segment overlap rules preserved, so that two Segments do not claim the same ink area.
42. As a researcher, I want Kraken Segment source metadata and Kraken ceiling preserved when present, so that future refinement remains possible.
43. As a researcher, I want Export to produce Processed line images and Line transcription files from current approved text, so that I can build training-ready data.
44. As a researcher, I want Export to warn about unpaired Segments and unused Text lines, so that I understand incomplete output.
45. As a researcher, I want Export to remain an annotation workflow, so that no separate export business object is introduced prematurely.
46. As a researcher, I want a Transcription PDF preview/share artifact, so that I can review Pairing results visually when useful.
47. As a frontend developer, I want OpenAPI-generated types in the merged frontend, so that API and UI stay aligned.
48. As a frontend developer, I want the platform shell with annote styling, so that authentication and Project navigation feel production-ready without losing editor identity.
49. As a backend developer, I want tests near the merged backend package, so that backend behavior is verified where it lives.
50. As a product owner, I want OCR prediction execution deferred, so that line/page/document OCR gets a proper design pass after the manual annotation model is stable.
51. As an agent working in parallel, I want vertical-slice issues with dependencies, so that independent implementation lanes can proceed safely.

## Implementation Decisions

- Annote is the production app root. The existing platform backend, frontend, and infrastructure are merged into annote, then duplicate root app folders are removed after functionality is carried over.
- The root model workspace remains at repository root permanently. Production app code may depend on packaged inference behavior, but research, training, notebooks, datasets, and checkpoints do not move into annote.
- Infrastructure lives beside backend and frontend under the annote app root so database migrations and future Terraform remain operationally separated from backend code.
- The backend uses DDD bounded contexts. Users owns authentication and identity. Project owns workspaces and sharing. Document owns Document, Document part, Line, Transcription, Line transcription, media, and access to manuscript structure. Annotation owns editor workflows: Pairing progress, Review status, Annotation history, Export, and restore. Inference exists but OCR execution design is deferred.
- There is no separate Segment table. Segment is an editor-facing term for the geometry persisted as a Line.
- Transcription is document-level. Ground truth is a document-level layer; Line transcription is the canonical per-Line approved text within that layer.
- Page transcription is an import and Pairing helper only. It may be partial and is not canonical until a human accepts or edits text into Line transcription.
- Review status belongs to a Document part/Page as a boolean. The frontend renders it as Reviewed or Unreviewed. Pairing progress remains a separate coverage metric.
- There is no Page lock. Human edits remain allowed, and Review status is human-controlled.
- Annotation history stores compact restorable Page annotation states. It excludes images, generated exports, and raw edit-by-edit events.
- Export remains an annotation application service for now. It produces training-ready artifacts from current Line geometry and approved Line transcription text without introducing a durable Export business object.
- OCR prediction and job execution are out of the first implementation slice and require a later focused design.
- The merged frontend uses the platform frontend stack and routing patterns while preserving annote's editor theme and workflow.
- Backend-specific tests move with the merged backend. Root-level tests remain only for root model work or true whole-repo integration.

## Testing Decisions

- Tests should verify external behavior at API, service, and UI seams rather than internal implementation details.
- The highest backend seam is authenticated API behavior through the merged FastAPI app, using database-backed tests for auth, Project access, Document parts, Lines, Line transcriptions, Review status, Annotation history, and Export.
- The highest frontend seam is user-visible workflow testing: login/project navigation, opening a Document part/Page, drawing or editing a Segment, pairing text, seeing Pairing progress, toggling Review status, restoring history, and exporting.
- Existing root platform auth/project/document tests are prior art for access-controlled API tests.
- Existing annote editor and service tests are prior art for canvas behavior, Pairing progress, page import, Annotation history, Export, PDF, and geometry rules.
- OpenAPI generation and TypeScript type checks should remain part of build verification so the merged frontend stays aligned with backend contracts.
- Export tests should assert generated artifact names, warnings, and content behavior without relying on the existing data folder.
- Annotation history tests should assert compact restore behavior and retention rules through public API/service seams.

## Out of Scope

- Designing or implementing OCR prediction execution for line, Page, or Document scope.
- Moving the root model workspace into annote.
- Migrating existing local data folder contents.
- Introducing a durable Export business object.
- Reintroducing Page lock or automatic review invalidation.
- Adding full event sourcing for annotation history.
- Reworking model training pipelines, notebooks, or checkpoints.
- Public publishing changes beyond what is needed to keep the merged platform coherent.

## Further Notes

The first implementation goal is a manual annotation vertical slice inside the authenticated Project and Document hierarchy. OCR prediction should be revisited after this slice proves the canonical document and annotation model.
