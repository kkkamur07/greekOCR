# Annote is the production platform root

Annote started as a standalone filesystem-backed annotation app, while the repository root already had the production platform spine for authentication, users, projects, documents, jobs, OpenAPI, Postgres, and migrations. We decided the final production app should live under `annote/`: move the root platform backend and frontend into `annote/`, then port annote's annotation workflows into that DDD structure instead of keeping annote as a separate app or incrementally copying auth features into the old standalone backend.

The canonical manuscript model belongs to the document context: `Document`, `DocumentPart`, `Line`, `Transcription`, and `LineTranscription`. Editor recovery workflows such as annotation history belong to a separate annotation context that references document parts and stores compact restorable annotation state.

The merged production layout should remove the duplicate root `backend/`, `frontend/`, and `infrastructure/` once their functionality is carried into `annote/`. Backend and frontend code live under `annote/backend/` and `annote/frontend/`; shared deployment and database infrastructure live under `annote/infrastructure/` so future Terraform can sit beside Alembic and related operational assets.

The root `model/` directory remains at repository root permanently because research, training experiments, checkpoints, and notebooks have a different lifecycle from the production app. Annote may depend on packaged inference behavior, but it should not absorb the full modeling workspace.
