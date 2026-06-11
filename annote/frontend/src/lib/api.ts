import type {
  AutoSegmentRequest,
  ExportProgressEvent,
  ExportResponse,
  HistoryListResponse,
  OcrProgressEvent,
  OcrResult,
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

export function pageImageUrl(stem: string, cacheKey?: string | null): string {
  const base = `${PUBLIC_API_BASE}/pages/${encodeURIComponent(stem)}/image`;
  if (!cacheKey) return base;
  return `${base}?t=${encodeURIComponent(cacheKey)}`;
}

export function segmentPreviewUrl(stem: string, segmentId: string): string {
  return `${PUBLIC_API_BASE}/pages/${encodeURIComponent(stem)}/segments/${encodeURIComponent(segmentId)}/preview`;
}

export function transcriptionPdfUrl(stem: string): string {
  return `${PUBLIC_API_BASE}/pages/${encodeURIComponent(stem)}/transcription.pdf`;
}

export function transcriptionSharePdfUrl(stem: string): string {
  return `${PUBLIC_API_BASE}/pages/${encodeURIComponent(stem)}/transcription.share.pdf`;
}

export async function fetchTranscriptionPdfBlob(
  stem: string,
  mode: "preview" | "share",
): Promise<Blob> {
  const base = mode === "preview" ? transcriptionPdfUrl(stem) : transcriptionSharePdfUrl(stem);
  const response = await fetch(`${base}?t=${Date.now()}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `PDF request failed: ${response.status}`);
  }
  return response.blob();
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

export async function lockPage(stem: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/lock`, { method: "POST" });
}

export async function unlockPage(stem: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/unlock`, { method: "POST" });
}

export async function fetchHistory(stem: string): Promise<HistoryListResponse> {
  return request<HistoryListResponse>(`/pages/${encodeURIComponent(stem)}/history`);
}

export async function restoreHistorySnapshot(stem: string, snapshotId: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(
    `/pages/${encodeURIComponent(stem)}/history/${encodeURIComponent(snapshotId)}/restore`,
    { method: "POST" },
  );
}

export async function autoSegmentPage(
  stem: string,
  options?: AutoSegmentRequest,
): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/segment`, {
    method: "POST",
    body: JSON.stringify(options ?? {}),
  });
}

export async function binarizePage(stem: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/binarize`, {
    method: "POST",
  });
}

export async function clearBinarizedPage(stem: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(`/pages/${encodeURIComponent(stem)}/binarize`, {
    method: "DELETE",
  });
}

type ExportStreamEvent =
  | ExportProgressEvent
  | { type: "done"; result: ExportResponse }
  | { type: "error"; detail: string };

export async function exportPage(
  stem: string,
  onProgress?: (event: ExportProgressEvent) => void,
): Promise<ExportResponse> {
  const response = await fetch(`${apiBase()}/pages/${encodeURIComponent(stem)}/export/stream`, {
    method: "POST",
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

export async function runSegmentOcr(stem: string, segmentId: string): Promise<PageAnnotation> {
  return request<PageAnnotation>(
    `/pages/${encodeURIComponent(stem)}/segments/${encodeURIComponent(segmentId)}/ocr`,
    { method: "POST" },
  );
}

type OcrStreamEvent =
  | OcrProgressEvent
  | { type: "done"; result: OcrResult }
  | { type: "error"; detail: string };

export async function ocrPage(
  stem: string,
  onProgress?: (event: OcrProgressEvent) => void,
): Promise<OcrResult> {
  const response = await fetch(`${apiBase()}/pages/${encodeURIComponent(stem)}/ocr/stream`, {
    method: "POST",
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `OCR failed: ${response.status}`);
  }
  if (!response.body) {
    throw new Error("OCR failed: empty response");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: OcrResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as OcrStreamEvent;
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
    const event = JSON.parse(buffer) as OcrStreamEvent;
    if (event.type === "progress") {
      onProgress?.(event);
    } else if (event.type === "done") {
      result = event.result;
    } else if (event.type === "error") {
      throw new Error(event.detail);
    }
  }

  if (!result) {
    throw new Error("OCR incomplete");
  }
  return result;
}
