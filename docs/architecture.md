# Nomicous technical architecture

Nomicous separates the browser editor, platform API, persistence, and
CPU-intensive inference. The same workflow can use local inference, optional
remote inference, or a future model host.

## System overview

```mermaid
flowchart LR
    Researcher["Researcher"] --> Browser["Next.js editor"]
    Browser -->|"HTTPS JSON + JWT/session"| API["FastAPI platform API"]
    API --> DB[("Postgres")]
    API --> Storage[("Private page storage")]
    API --> Jobs["Durable platform jobs"]
    Jobs --> PlatformWorker["Platform worker"]
    PlatformWorker --> InferenceAPI["Inference API"]
    InferenceAPI --> InferenceJobs[("Inference job queue")]
    InferenceJobs --> InferenceWorker["Inference worker"]
    InferenceWorker --> Models["BLLA ONNX + Calamari ONNX"]
    InferenceWorker -->|"signed callback"| API
    Browser -->|"loopback HTTP"| Helper["Local inference helper"]
    Helper --> LocalModels["Local CPU model cache"]
    Browser -->|"persist local result"| API
    API -.-> Vercel["Vercel"]
    DB -.-> Supabase["Supabase Postgres"]
    Storage -.-> SupabaseStorage["Supabase Storage"]
    InferenceAPI -.-> Docker["Persistent Docker host"]
```

Local inference runs on the researcher’s machine through `127.0.0.1:8001`.
Remote inference creates a durable platform job, dispatches it through
persistent workers, and merges the signed callback into the platform state.
Local inference avoids a separate cloud ML service; the page still resides in
the configured platform storage.

## Stack choices

- **Next.js and React:** productive routing, standalone builds, and a
  responsive browser editor for annotation, pairing, review, and jobs.
- **FastAPI:** typed Python API contracts shared naturally with the inference
  and research code; bounded contexts cover users, projects, documents,
  annotations, ML, and jobs.
- **Postgres:** transactional source of truth for users, projects, sharing,
  annotations, transcription layers, model bindings, and durable jobs.
- **Supabase:** managed Postgres and private Storage. The browser does not use
  Supabase Auth, PostgREST, Realtime, Edge Functions, or direct Storage access.
- **Vercel:** suitable for the landing page, Next.js editor, and
  request/response API, but not long-running PyTorch workers.
- **Docker:** repeatable local packaging and persistent worker deployment.

## Annotation and sharing

```text
User
 └── Project
      ├── shared users
      └── Documents
            ├── Document parts (pages)
            │     ├── Blocks and Segments
            │     ├── page transcription lines
            │     └── review state and history
            └── Transcription layers
                  ├── model transcription
                  └── ground-truth transcription
```

A Segment is a user-drawn or model-created region for one written line. A
researcher may accept, edit, or ignore a Model transcription. It becomes
Ground truth only after that human decision. Paired segments can be exported
as processed line images and text files. Public documents use separate public
routes; draft documents remain protected.

Sharing is represented by project membership records and enforced by FastAPI
authorization. The platform does not automatically pair, approve, or publish
model output.

## Jobs and callbacks

```mermaid
sequenceDiagram
    participant UI as Browser
    participant API as PlatformAPI
    participant DB as Postgres
    participant PW as PlatformWorker
    participant IA as InferenceAPI
    participant IW as InferenceWorker

    UI->>API: Create segment or transcribe job
    API->>DB: Insert jobs(status=pending)
    API-->>UI: Return product job id
    PW->>DB: Claim platform job
    PW->>IA: Submit image, model, and params
    IA->>DB: Insert inference_jobs(status=pending)
    IA-->>PW: Return inference job id
    PW->>DB: Set platform job to waiting
    IW->>DB: Claim inference job
    IW->>IW: Run model
    IW->>DB: Store done or failed output
    IW->>API: Signed completion callback
    API->>DB: Lock, merge, and finalize product job
```

The platform job is user-visible. `pending` means unclaimed, `running` means
the platform worker is processing it, `waiting` means inference accepted it,
and `done` or `failed` are terminal. Callback locking and terminal-state
checks make retries idempotent.

## Job notifications

Nomicous does not currently provide email, push, SMS, or third-party
notifications. Job progress uses Postgres `NOTIFY`, an API-local SSE fan-out,
and polling fallback:

```mermaid
flowchart LR
    Worker["Worker or callback"] --> Commit["Commit job status"]
    Commit --> DB[("Postgres")]
    DB --> Notify["NOTIFY platform_jobs"]
    Notify --> Listener["Dedicated LISTEN connection"]
    Listener --> Fanout["API fan-out"]
    Fanout --> SSE["SSE job events"]
    SSE --> Browser["Browser UI"]
    Browser -.-> Poll["GET job polling fallback"]
    Poll --> API["Platform API"]
    API --> DB
```

Postgres remains authoritative. The state change commits first, then
`NOTIFY` wakes the listener. SSE reloads the authorized job before sending it.
If SSE is unavailable or idle, the frontend polls `GET /jobs/{id}`. Vercel
production disables long-lived listeners, so polling is the expected hosted
fallback.

## Local helper

The helper is a small FastAPI sidecar with no database, platform queue,
project authorization, or storage credentials:

```mermaid
sequenceDiagram
    participant B as Browser
    participant API as PlatformAPI
    participant H as LocalHelper
    participant M as LocalModels

    B->>API: Fetch authorized page image
    API-->>B: Image bytes
    B->>H: POST /inference/v1/run
    H->>M: Load cached model
    M-->>H: Segments or transcription
    H-->>B: Inference result
    B->>API: Persist authenticated result
    API-->>B: Updated page and job
```

It synchronizes the public registry with ETags, downloads weights lazily,
defaults to loopback, and caches weights at `~/.nomicous/hf/cache`. Exposing it
outside loopback requires secure mode, a strong helper secret, and TLS.

## Security boundaries

- Authentication is application-owned: password hashes, rotating sessions,
  JWT access tokens, and CSRF protection are implemented by FastAPI.
- The browser never connects directly to Postgres or private Storage.
- The API checks ownership or sharing before returning documents, images,
  annotations, and jobs.
- Inference workers receive only inference and callback credentials, not
  migration credentials or platform JWT secrets.
- Registry artifacts use pinned revisions and SHA-256 verification where
  configured.

For database roles, pooling, migrations, state machines, and callback
idempotency, see [`database-design.md`](database-design.md).
