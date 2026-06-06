import type {
  ExportProgressEvent,
  ExportResponse,
  PageAnnotation,
  PageListResponse,
  PageSummary,
  TranscriptionResponse,
} from "@/types/api";

const PUBLIC_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:5050";

/** Browser calls the published host; SSR inside Docker uses the compose service name. */
function apiBase(): string {
  if (typeof window !== "undefined") {
    return PUBLIC_API_BASE;
  }
  return process.env.API_INTERNAL_URL ?? PUBLIC_API_BASE;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function pageImageUrl(stem: string): string {
  return `${PUBLIC_API_BASE}/pages/${encodeURIComponent(stem)}/image`;
}

export async function fetchPages(): Promise<PageListResponse> {
  return request<PageListResponse>("/pages");
}

export async function importPage(image: File, transcription?: File): Promise<PageSummary> {
  const form = new FormData();
  form.append("image", image);
  if (transcription) form.append("transcription", transcription);

  const response = await fetch(`${apiBase()}/pages/import`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Import failed: ${response.status}`);
  }
  return response.json() as Promise<PageSummary>;
}

export async function fetchTranscription(stem: string): Promise<TranscriptionResponse> {
  return request<TranscriptionResponse>(`/pages/${encodeURIComponent(stem)}/transcription`);
}

export async function fetchAnnotation(stem: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/annotation`);
}

export async function saveAnnotation(stem: string, annotation: PageAnnotation): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/annotation`, {
    method: "PUT",
    body: JSON.stringify(annotation),
  });
}

type ExportStreamEvent =
  | ExportProgressEvent
  | { type: "done"; result: ExportResponse }
  | { type: "error"; detail: string };

export async function exportPage(
  stem: string,
  options?: { binarize?: boolean },
  onProgress?: (event: ExportProgressEvent) => void,
): Promise<ExportResponse> {
  const response = await fetch(`${apiBase()}/pages/${encodeURIComponent(stem)}/export/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options ?? {}),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Export failed: ${response.status}`);
  }
  if (!response.body) {
    throw new Error("Export failed: empty response");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ExportResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as ExportStreamEvent;
      if (event.type === "progress") {
        onProgress?.(event);
      } else if (event.type === "done") {
        result = event.result;
      } else if (event.type === "error") {
        throw new Error(event.detail);
      }
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer) as ExportStreamEvent;
    if (event.type === "progress") {
      onProgress?.(event);
    } else if (event.type === "done") {
      result = event.result;
    } else if (event.type === "error") {
      throw new Error(event.detail);
    }
  }

  if (!result) {
    throw new Error("Export incomplete");
  }
  return result;
}
