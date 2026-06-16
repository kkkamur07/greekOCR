/**
 * Platform API client — JWT from localStorage, OpenAPI-aligned types.
 * Regenerate types: `npm run codegen:api` (after `python scripts/export_openapi.py`).
 */
import { clearAccessToken, getAccessToken } from '../auth/storage';
import { ApiError, parseApiError } from './errors';
import type { components } from './schema';

export type TokenResponse = components['schemas']['TokenResponse'];
export type UserResponse = components['schemas']['UserResponse'];
export type LoginRequest = components['schemas']['LoginRequest'];
export type RegisterRequest = components['schemas']['RegisterRequest'];
export type ProjectResponse = components['schemas']['ProjectResponse'];
export type ProjectCreateRequest = components['schemas']['ProjectCreateRequest'];
export type DocumentResponse = components['schemas']['DocumentResponse'];
export type DocumentWithPartsResponse = components['schemas']['DocumentWithPartsResponse'];
export type DocumentCreateRequest = components['schemas']['DocumentCreateRequest'];
export type DocumentPartResponse = components['schemas']['DocumentPartResponse'];
export type DocumentWorkflow = components['schemas']['DocumentWorkflow'];
export type ReorderPartsRequest = components['schemas']['ReorderPartsRequest'];
export type PublicLayoutResponse = components['schemas']['PublicLayoutResponse'];
export type PublicTranscriptionLayerResponse =
  components['schemas']['PublicTranscriptionLayerResponse'];
export type JobResponse = components['schemas']['JobResponse'];
export type JobStatus = components['schemas']['JobStatus'];
export type EnqueueTestJobRequest = components['schemas']['EnqueueTestJobRequest'];
export type EnqueueTestJobResponse = components['schemas']['EnqueueTestJobResponse'];
export type LayoutPoint = [number, number];
export type LayoutBlockResponse = {
  id: string;
  box?: LayoutPoint[] | null;
  manual_geometry?: boolean | null;
};
export type LayoutLineResponse = {
  id: string;
  block_id?: string | null;
  baseline?: LayoutPoint[] | null;
  mask?: LayoutPoint[] | null;
  manual_geometry?: boolean | null;
};
export type PartLayoutResponse = {
  blocks: LayoutBlockResponse[];
  lines: LayoutLineResponse[];
};
export type LineGeometryKind = 'polygon' | 'rectangle';
export type LineSource = 'manual' | 'kraken' | 'model';
export type LinePoint = [number, number];
export type LineResponse = {
  id: string;
  part_id: string;
  block_id: string | null;
  order: number;
  kind: LineGeometryKind;
  points: LinePoint[];
  source: LineSource;
  source_metadata: Record<string, unknown> | null;
  kraken_ceiling: LinePoint[] | null;
  manual_geometry: boolean;
  line_transcriptions: unknown[];
  created_at: string;
};
export type LineUpsertRequest = {
  id?: string;
  order: number;
  kind: LineGeometryKind;
  points: LinePoint[];
  source: LineSource;
};
export type LinesReplaceRequest = {
  lines: LineUpsertRequest[];
};
export type UpdateLineGeometryRequest = {
  baseline?: LayoutPoint[] | null;
  mask?: LayoutPoint[] | null;
};
export type ResetPartLayoutRequest = {
  line_ids?: string[] | null;
};

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ||
  'http://localhost:8000';

function redirectToLogin(): void {
  clearAccessToken();
  const loginPath = '/login';
  if (window.location.pathname !== loginPath && window.location.pathname !== '/register') {
    window.location.assign(loginPath);
  }
}

type RequestOptions = Omit<RequestInit, 'body'> & {
  body?: unknown;
  skipAuth?: boolean;
};

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, skipAuth, headers: initHeaders, ...rest } = options;
  const headers = new Headers(initHeaders);

  if (body !== undefined && !(body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  if (!skipAuth) {
    const token = getAccessToken();
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    headers,
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });

  if (response.status === 401 && !skipAuth) {
    redirectToLogin();
    throw new ApiError('Unauthorized', 401);
  }

  if (!response.ok) {
    throw await parseApiError(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function mediaUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}

export function publicPartMediaUrl(partId: string): string {
  return `${API_BASE_URL}/public/media/parts/${partId}`;
}

export const api = {
  login: (body: LoginRequest) =>
    apiRequest<TokenResponse>('/auth/login', { method: 'POST', body, skipAuth: true }),

  register: (body: RegisterRequest) =>
    apiRequest<TokenResponse>('/auth/register', { method: 'POST', body, skipAuth: true }),

  me: () => apiRequest<UserResponse>('/me'),

  listProjects: () => apiRequest<ProjectResponse[]>('/projects'),

  createProject: (body: ProjectCreateRequest) =>
    apiRequest<ProjectResponse>('/projects', { method: 'POST', body }),

  getProject: (projectId: string) =>
    apiRequest<ProjectResponse>(`/projects/${projectId}`),

  listDocuments: (projectId: string, includeArchived = false) =>
    apiRequest<DocumentResponse[]>(
      `/projects/${projectId}/documents?include_archived=${includeArchived}`,
    ),

  createDocument: (projectId: string, body: DocumentCreateRequest) =>
    apiRequest<DocumentResponse>(`/projects/${projectId}/documents`, {
      method: 'POST',
      body,
    }),

  getDocument: (projectId: string, documentId: string) =>
    apiRequest<DocumentWithPartsResponse>(
      `/projects/${projectId}/documents/${documentId}`,
    ),

  uploadPart: (projectId: string, documentId: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    return apiRequest<DocumentPartResponse>(
      `/projects/${projectId}/documents/${documentId}/parts`,
      { method: 'POST', body: form },
    );
  },

  reorderParts: (projectId: string, documentId: string, body: ReorderPartsRequest) =>
    apiRequest<DocumentPartResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/reorder`,
      { method: 'PATCH', body },
    ),

  deletePart: (projectId: string, documentId: string, partId: string) =>
    apiRequest<void>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}`,
      { method: 'DELETE' },
    ),

  getPartLayout: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PartLayoutResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout`,
    ),

  listPartLines: (projectId: string, documentId: string, partId: string) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
    ),

  replacePartLines: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LinesReplaceRequest,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
      { method: 'PUT', body },
    ),

  updateLineGeometry: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: UpdateLineGeometryRequest,
  ) =>
    apiRequest<LayoutLineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout/lines/${lineId}`,
      { method: 'PATCH', body },
    ),

  resetPartLayout: (
    projectId: string,
    documentId: string,
    partId: string,
    body: ResetPartLayoutRequest,
  ) =>
    apiRequest<PartLayoutResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout/reset`,
      { method: 'POST', body },
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

  enqueueTestJob: (body: EnqueueTestJobRequest = { handler: 'noop' }) =>
    apiRequest<EnqueueTestJobResponse>('/jobs/test', { method: 'POST', body }),

  getJob: (jobId: string) => apiRequest<JobResponse>(`/jobs/${jobId}`),
};
