/**
 * Platform API client - short-lived JWT in memory, OpenAPI-aligned types.
 * Regenerate types: `npm run codegen:api` (after `python scripts/platform/export_openapi.py`).
 */
import { redirectToLogin } from "../auth/session";
import { getAccessToken, setAccessToken } from "../auth/storage";
import { JOB_WAIT_POLL_INTERVAL_MS } from "../utils/jobPolling";
import { waitForSubscribedJob } from "../utils/jobSubscription";
import {
  collectCursorPages,
  type CursorPage,
  type CursorPageOptions,
} from "./cursorPagination";
import { ApiError, parseApiError } from "./errors";
import { dedupedGet, getAuthRequestVersion } from "./getCache";
import type { components } from "./schema";

export type TokenResponse = components["schemas"]["TokenResponse"];
export type UserResponse = components["schemas"]["UserResponse"];
export type LoginRequest = components["schemas"]["LoginRequest"];
export type RegisterRequest = components["schemas"]["RegisterRequest"];
export type ProjectResponse = components["schemas"]["ProjectResponse"];
export type ProjectCreateRequest =
  components["schemas"]["ProjectCreateRequest"];
export type ProjectUpdateRequest =
  components["schemas"]["ProjectUpdateRequest"];
export type DocumentResponse = components["schemas"]["DocumentResponse"];
export type DocumentWithPartsResponse =
  components["schemas"]["DocumentWithPartsResponse"];
export type DocumentCreateRequest =
  components["schemas"]["DocumentCreateRequest"];
export type DocumentUpdateRequest =
  components["schemas"]["DocumentUpdateRequest"];
export type DocumentPartResponse =
  components["schemas"]["DocumentPartResponse"];
export type DocumentPartUpdateRequest =
  components["schemas"]["DocumentPartUpdateRequest"];
export type DocumentWorkflow = components["schemas"]["DocumentWorkflow"];
export type ReorderPartsRequest = components["schemas"]["ReorderPartsRequest"];
export type PublicLayoutResponse =
  components["schemas"]["PublicLayoutResponse"];
export type PublicLineResponse = NonNullable<
  PublicLayoutResponse["lines"]
>[number];
export type PublicTranscriptionLayerResponse =
  components["schemas"]["PublicTranscriptionLayerResponse"];
export type TranscriptionLayerResponse =
  components["schemas"]["TranscriptionLayerResponse"];
export type LineTranscriptionResponse =
  components["schemas"]["LineTranscriptionResponse"] & {
    text_source?: string | null;
    character_confidences?:
      components["schemas"]["CharacterConfidence"][] | null;
  };
export type CharacterConfidence = components["schemas"]["CharacterConfidence"];
export type JobResponse = components["schemas"]["JobResponse"];
export type JobStatus = components["schemas"]["JobStatus"];
export type EnqueueJobResponse = components["schemas"]["EnqueueJobResponse"];
export type SegmentPartRequest = {
  use_otsu_refinement?: boolean;
  otsu_sphere_radius?: number;
  target_max_points?: number;
  min_iou?: number;
  min_area_ratio?: number;
  split_large_lines?: boolean;
  split_vertical_gap_px?: number;
};
export type InferenceModelResponse =
  components["schemas"]["InferenceModelResponse"];
export type InferenceTask = components["schemas"]["InferenceTask"];
export type ResolvedModelBindingResponse =
  components["schemas"]["ResolvedModelBindingResponse"];
export type EnqueueTestJobRequest =
  components["schemas"]["EnqueueTestJobRequest"];
export type EnqueueTestJobResponse =
  components["schemas"]["EnqueueTestJobResponse"];
export type LayoutPoint = [number, number];
export type GeometryValue =
  | LayoutPoint[]
  | {
      points?: LayoutPoint[];
      type?: string;
      coordinates?: LayoutPoint[];
    };
export type LayoutBlockResponse = {
  id: string;
  box?: GeometryValue | null;
  manual_geometry?: boolean | null;
};
export type LayoutLineResponse = {
  id: string;
  block_id?: string | null;
  baseline?: GeometryValue | null;
  mask?: GeometryValue | null;
  manual_geometry?: boolean | null;
};
export type PartLayoutResponse = {
  blocks: LayoutBlockResponse[];
  lines: LayoutLineResponse[];
};
export type LineGeometryKind = "polygon" | "rectangle";
export type LineSource = "manual" | "kraken" | "model";
export type LinePoint = [number, number];
export type LineResponse = {
  id: string;
  part_id: string;
  block_id: string | null;
  order: number;
  kind: LineGeometryKind;
  points: LinePoint[];
  baseline: GeometryValue;
  mask: GeometryValue | null;
  source: LineSource;
  source_metadata: Record<string, unknown> | null;
  kraken_ceiling: LinePoint[] | null;
  manual_geometry: boolean;
  line_transcriptions: LineTranscriptionResponse[];
  created_at: string;
};
export type LineCreateRequest = {
  order: number;
  kind: LineGeometryKind;
  points: LinePoint[];
  block_id?: string | null;
  baseline?: GeometryValue | null;
  mask?: GeometryValue | null;
};
export type LinePatchRequest = {
  order?: number;
  block_id?: string | null;
  baseline?: GeometryValue | null;
  mask?: GeometryValue | null;
  points?: LinePoint[];
};
export type LineUpsertRequest = {
  id?: string;
  order: number;
  kind: LineGeometryKind;
  points: LinePoint[];
  block_id?: string | null;
  source: LineSource;
  source_metadata?: Record<string, unknown> | null;
  kraken_ceiling?: LinePoint[] | null;
  baseline?: GeometryValue | null;
  mask?: GeometryValue | null;
  approved_text?: string | null;
};
export type LinesReplaceRequest = {
  lines: LineUpsertRequest[];
};
export type UpdateLineGeometryRequest = {
  baseline?: GeometryValue | null;
  mask?: GeometryValue | null;
};
export type ResetPartLayoutRequest = {
  line_ids?: string[] | null;
};
export type PageTranscriptionTextLineResponse = {
  order: number;
  text: string;
  paired_line_id: string | null;
};
export type PairingProgressResponse = {
  paired_lines: number;
  total_lines: number;
  percent: number;
};
export type PagePairingResponse = {
  text_lines: PageTranscriptionTextLineResponse[];
  pairing_progress: PairingProgressResponse;
};
export type PageTranscriptionImportRequest = {
  text: string;
};
export type PairTextLineRequest = {
  line_id: string;
  text_line_order: number;
};
export type LineTranscriptionPatchRequest = {
  text: string;
};
export type CopyToGroundTruthRequest =
  components["schemas"]["CopyToGroundTruthRequest"];
export type CopyToGroundTruthResponse =
  components["schemas"]["CopyToGroundTruthResponse"];

export type PageResponse<T> = CursorPage<T>;

export type ListPageOptions = CursorPageOptions;

function cursorQuery(
  options: CursorPageOptions,
  extra?: (params: URLSearchParams) => void,
): string {
  const params = new URLSearchParams();
  if (options.cursor) params.set("cursor", options.cursor);
  if (options.limit) params.set("limit", String(options.limit));
  extra?.(params);
  return params.toString();
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";
export const API_ORIGIN = new URL(API_BASE_URL).origin;
const CSRF_COOKIE_NAME =
  process.env.NEXT_PUBLIC_CSRF_COOKIE_NAME || "greekocr-csrf";

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
  skipAuth?: boolean;
  /** Bypass in-flight GET deduplication (default: dedupe GETs). */
  skipDedup?: boolean;
};

let refreshPromise: Promise<TokenResponse> | null = null;

function csrfToken(): string | null {
  if (typeof document === "undefined") return null;
  const encodedName = `${encodeURIComponent(CSRF_COOKIE_NAME)}=`;
  const cookie = document.cookie
    .split("; ")
    .find((item) => item.startsWith(encodedName));
  if (!cookie) return null;
  try {
    return decodeURIComponent(cookie.slice(encodedName.length));
  } catch {
    return null;
  }
}

function addCsrfHeader(headers: Headers, method: string): void {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(method)) return;
  const token = csrfToken();
  if (token) headers.set("X-CSRF-Token", token);
}

/**
 * Refresh the cookie-backed session once for all concurrent callers.
 * The refresh route deliberately bypasses auth recovery to avoid recursion.
 */
export function refreshAccessToken(): Promise<TokenResponse> {
  refreshPromise ??= apiRequest<TokenResponse>("/auth/refresh", {
    method: "POST",
    skipAuth: true,
  })
    .then((token) => {
      setAccessToken(token.access_token);
      return token;
    })
    .finally(() => {
      refreshPromise = null;
    });
  return refreshPromise;
}

/**
 * Fetch a protected resource, refreshing the access token once after a 401.
 * It is shared by API requests and the event-stream client so each path has
 * the same recovery and sign-out behavior.
 */
export async function fetchWithAuthRecovery(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  return fetchWithAuthRecoveryAttempt(input, init, false);
}

async function fetchWithAuthRecoveryAttempt(
  input: RequestInfo | URL,
  init: RequestInit,
  retried: boolean,
): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getAccessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(input, {
    ...init,
    credentials: init.credentials ?? "include",
    headers,
  });
  if (response.status !== 401) return response;

  if (retried) {
    redirectToLogin();
    throw new ApiError("Unauthorized", 401);
  }

  try {
    await refreshAccessToken();
  } catch {
    redirectToLogin();
    throw new ApiError("Unauthorized", 401);
  }

  return fetchWithAuthRecoveryAttempt(input, init, true);
}

async function requestApiResponse(
  path: string,
  options: RequestOptions,
): Promise<Response> {
  const { body, skipAuth, headers: initHeaders, ...rest } = options;
  const headers = new Headers(initHeaders);
  const method = (options.method ?? "GET").toUpperCase();

  if (body !== undefined && !(body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  addCsrfHeader(headers, method);

  const init: RequestInit = {
    ...rest,
    credentials: "include",
    headers,
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  };

  if (skipAuth) {
    return fetch(`${API_BASE_URL}${path}`, init);
  }
  return fetchWithAuthRecovery(`${API_BASE_URL}${path}`, init);
}

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  // Abort signals belong to one component lifecycle. Sharing an abortable
  // fetch lets one Strict Mode cleanup cancel a newer mount's request.
  if (method === "GET" && !options.skipDedup && !options.signal) {
    return dedupedGet(`auth:${getAuthRequestVersion()} GET ${path}`, () =>
      apiRequest<T>(path, { ...options, skipDedup: true }),
    );
  }

  const response = await requestApiResponse(path, options);

  if (!response.ok) {
    throw await parseApiError(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function fetchBinaryApi(
  path: string,
  options: RequestOptions = {},
): Promise<Blob> {
  const response = await requestApiResponse(path, options);

  if (!response.ok) {
    throw await parseApiError(response);
  }

  return response.blob();
}

export function publicPartMediaUrl(partId: string): string {
  return `${API_BASE_URL}/public/media/parts/${partId}`;
}

export type TranscribeJobResult = {
  transcription_id: string;
  lines: Array<{ line_id: string; text: string; confidence: number }>;
};

export const api = {
  login: (body: LoginRequest) =>
    apiRequest<TokenResponse>("/auth/login", {
      method: "POST",
      body,
      skipAuth: true,
    }),

  register: (body: RegisterRequest) =>
    apiRequest<TokenResponse>("/auth/register", {
      method: "POST",
      body,
      skipAuth: true,
    }),

  refresh: refreshAccessToken,

  logout: () =>
    apiRequest<void>("/auth/logout", { method: "POST", skipAuth: true }),

  me: () => apiRequest<UserResponse>("/me"),

  listProjectsPage: (options: ListPageOptions = {}) => {
    const query = cursorQuery(options);
    return apiRequest<PageResponse<ProjectResponse>>(
      query ? `/projects?${query}` : "/projects",
      {
        signal: options.signal,
      },
    );
  },

  listProjects: (options?: { maxPages?: number; signal?: AbortSignal }) =>
    collectCursorPages(
      (pageOptions) => api.listProjectsPage(pageOptions),
      options,
    ),

  createProject: (body: ProjectCreateRequest) =>
    apiRequest<ProjectResponse>("/projects", { method: "POST", body }),

  getProject: (projectId: string) =>
    apiRequest<ProjectResponse>(`/projects/${projectId}`),

  updateProject: (projectId: string, body: ProjectUpdateRequest) =>
    apiRequest<ProjectResponse>(`/projects/${projectId}`, {
      method: "PATCH",
      body,
    }),

  deleteProject: (projectId: string) =>
    apiRequest<void>(`/projects/${projectId}`, { method: "DELETE" }),

  listDocumentsPage: (
    projectId: string,
    includeArchived = false,
    options: ListPageOptions = {},
  ) => {
    const query = cursorQuery(options, (params) =>
      params.set("include_archived", String(includeArchived)),
    );
    return apiRequest<PageResponse<DocumentResponse>>(
      `/projects/${projectId}/documents?${query}`,
      { signal: options.signal },
    );
  },

  listDocuments: (
    projectId: string,
    includeArchived = false,
    options?: { maxPages?: number; signal?: AbortSignal },
  ) =>
    collectCursorPages(
      (pageOptions) =>
        api.listDocumentsPage(projectId, includeArchived, pageOptions),
      options,
    ),

  createDocument: (projectId: string, body: DocumentCreateRequest) =>
    apiRequest<DocumentResponse>(`/projects/${projectId}/documents`, {
      method: "POST",
      body,
    }),

  getDocument: (projectId: string, documentId: string) =>
    apiRequest<DocumentWithPartsResponse>(
      `/projects/${projectId}/documents/${documentId}`,
    ),

  updateDocument: (
    projectId: string,
    documentId: string,
    body: DocumentUpdateRequest,
  ) =>
    apiRequest<DocumentResponse>(
      `/projects/${projectId}/documents/${documentId}`,
      {
        method: "PATCH",
        body,
      },
    ),

  deleteDocument: (projectId: string, documentId: string) =>
    apiRequest<void>(`/projects/${projectId}/documents/${documentId}`, {
      method: "DELETE",
    }),

  listTranscriptions: (projectId: string, documentId: string) =>
    apiRequest<TranscriptionLayerResponse[]>(
      `/projects/${projectId}/documents/${documentId}/transcriptions`,
    ),

  uploadPart: (projectId: string, documentId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest<DocumentPartResponse>(
      `/projects/${projectId}/documents/${documentId}/parts`,
      { method: "POST", body: form },
    );
  },

  reorderParts: (
    projectId: string,
    documentId: string,
    body: ReorderPartsRequest,
  ) =>
    apiRequest<DocumentPartResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/reorder`,
      { method: "PATCH", body },
    ),

  deletePart: (projectId: string, documentId: string, partId: string) =>
    apiRequest<void>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}`,
      { method: "DELETE" },
    ),

  updatePartReviewStatus: (
    projectId: string,
    documentId: string,
    partId: string,
    body: DocumentPartUpdateRequest,
  ) =>
    apiRequest<DocumentPartResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}`,
      { method: "PATCH", body },
    ),

  getPartLayout: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PartLayoutResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout`,
    ),

  listPartLines: (projectId: string, documentId: string, partId: string) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
    ),

  createPartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LineCreateRequest,
  ) =>
    apiRequest<LineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
      { method: "POST", body },
    ),

  replacePartLines: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LinesReplaceRequest,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
      { method: "PUT", body },
    ),

  deletePartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<void>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines/${lineId}`,
      { method: "DELETE" },
    ),

  patchPartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: LinePatchRequest,
  ) =>
    apiRequest<LineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines/${lineId}`,
      { method: "PATCH", body },
    ),

  updateLineGeometry: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: UpdateLineGeometryRequest,
  ) =>
    apiRequest<LineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines/${lineId}`,
      { method: "PATCH", body },
    ),

  resetPartLayout: (
    projectId: string,
    documentId: string,
    partId: string,
    body: ResetPartLayoutRequest,
  ) =>
    apiRequest<PartLayoutResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout/reset`,
      { method: "POST", body },
    ),

  getPagePairing: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PagePairingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/pairing`,
    ),

  importPageTranscription: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PageTranscriptionImportRequest,
  ) =>
    apiRequest<PagePairingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/page-transcription`,
      { method: "PUT", body },
    ),

  pairTextLine: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PairTextLineRequest,
  ) =>
    apiRequest<PagePairingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/pairings`,
      { method: "POST", body },
    ),

  generateTranscriptionPdf: (
    projectId: string,
    documentId: string,
    partId: string,
  ) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/transcription-pdf`,
      { method: "POST" },
    ),

  segmentPart: (
    projectId: string,
    documentId: string,
    partId: string,
    body?: SegmentPartRequest,
  ) =>
    apiRequest<EnqueueJobResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment`,
      { method: "POST", body: body ?? {} },
    ),

  enqueueTranscribePart: (
    projectId: string,
    documentId: string,
    partId: string,
    body?: { model_id?: string; line_ids?: string[] },
  ) =>
    apiRequest<EnqueueJobResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/transcribe`,
      { method: "POST", body: body ?? {} },
    ),

  persistLocalTranscribe: (
    projectId: string,
    documentId: string,
    partId: string,
    body: {
      registry_model_id: string;
      registry_tag?: string;
      lines: Array<{
        line_id: string;
        text: string;
        confidence: number;
        character_confidences?: Array<{ char: string; confidence: number }>;
      }>;
    },
  ) =>
    apiRequest<{
      job_id: string;
      transcription_id: string;
      lines: Array<{ line_id: string; text: string; confidence: number }>;
    }>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/local-inference/transcribe`,
      { method: "POST", body },
    ),

  persistLocalSegment: (
    projectId: string,
    documentId: string,
    partId: string,
    body: {
      registry_model_id: string;
      registry_tag?: string;
      output: Record<string, unknown>;
    },
  ) =>
    apiRequest<{
      job_id: string;
      blocks_count: number;
      lines_count: number;
      added_lines: number;
      pruned_lines: number;
      preserved_manual_lines: number;
    }>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/local-inference/segment`,
      { method: "POST", body },
    ),

  listInferenceModels: () =>
    apiRequest<InferenceModelResponse[]>("/inference/models"),

  resolvePartModelBinding: (
    projectId: string,
    documentId: string,
    partId: string,
    task: InferenceTask,
  ) =>
    apiRequest<ResolvedModelBindingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/model-bindings/resolve?task=${task}`,
    ),

  updateGroundTruthLineText: (
    projectId: string,
    documentId: string,
    transcriptionId: string,
    lineId: string,
    body: LineTranscriptionPatchRequest,
  ) =>
    apiRequest<LineTranscriptionResponse>(
      `/projects/${projectId}/documents/${documentId}/transcriptions/${transcriptionId}/lines/${lineId}`,
      { method: "PATCH", body },
    ),

  copyToGroundTruth: (
    projectId: string,
    documentId: string,
    transcriptionId: string,
    body: CopyToGroundTruthRequest,
  ) =>
    apiRequest<CopyToGroundTruthResponse>(
      `/projects/${projectId}/documents/${documentId}/transcriptions/${transcriptionId}/copy-to-ground-truth`,
      { method: "POST", body },
    ),

  getPublicDocument: (projectId: string, documentId: string) =>
    apiRequest<DocumentWithPartsResponse>(
      `/public/projects/${projectId}/documents/${documentId}`,
      { skipAuth: true },
    ),

  getPublicLayout: (projectId: string, documentId: string) =>
    apiRequest<PublicLayoutResponse>(
      `/public/projects/${projectId}/documents/${documentId}/layout`,
      { skipAuth: true },
    ),

  listPublicTranscriptions: (projectId: string, documentId: string) =>
    apiRequest<PublicTranscriptionLayerResponse[]>(
      `/public/projects/${projectId}/documents/${documentId}/transcriptions`,
      { skipAuth: true },
    ),

  getPublicTranscriptionPdf: (
    projectId: string,
    documentId: string,
    partId: string,
  ) =>
    fetchBinaryApi(
      `/public/projects/${projectId}/documents/${documentId}/parts/${partId}/transcription-pdf`,
      { skipAuth: true },
    ),

  getPublicPageXml: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(
      `/public/projects/${projectId}/documents/${documentId}/parts/${partId}/page-xml`,
      { skipAuth: true },
    ),

  enqueueTestJob: (body: EnqueueTestJobRequest = { handler: "noop" }) =>
    apiRequest<EnqueueTestJobResponse>("/jobs/test", { method: "POST", body }),

  getJob: (jobId: string) => apiRequest<JobResponse>(`/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    apiRequest<JobResponse>(`/jobs/${jobId}/cancel`, { method: "POST" }),

  listProjectJobsPage: (projectId: string, options: ListPageOptions = {}) => {
    const query = cursorQuery(options);
    return apiRequest<PageResponse<JobResponse>>(
      `/projects/${projectId}/jobs${query ? `?${query}` : ""}`,
      { signal: options.signal },
    );
  },

  listProjectJobs: (
    projectId: string,
    options?: { maxPages?: number; signal?: AbortSignal },
  ) =>
    collectCursorPages(
      (pageOptions) => api.listProjectJobsPage(projectId, pageOptions),
      options,
    ),
};

export async function waitForJob(
  jobId: string,
  options?: { timeoutMs?: number; onUpdate?: (job: JobResponse) => void },
): Promise<JobResponse> {
  return waitForSubscribedJob(jobId, {
    ...options,
    eventsUrl: `${API_BASE_URL}/jobs/${jobId}/events`,
    token: getAccessToken(),
    getJob: api.getJob,
    intervalMs: JOB_WAIT_POLL_INTERVAL_MS,
  });
}
