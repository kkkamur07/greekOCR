"use client";

import { useEffect, useState } from "react";
import { fetchTranscriptionPdfBlob, transcriptionPdfUrl, transcriptionSharePdfUrl } from "@/lib/api";

export type TranscriptionPdfMode = "preview" | "share";

interface TranscriptionPdfPanelProps {
  stem: string;
  mode: TranscriptionPdfMode;
  locked: boolean;
  refreshKey: number;
  onClose: () => void;
  onSwitchMode: (mode: TranscriptionPdfMode) => void;
}

export default function TranscriptionPdfPanel({
  stem,
  mode,
  locked,
  refreshKey,
  onClose,
  onSwitchMode,
}: TranscriptionPdfPanelProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    async function loadPdf() {
      setLoading(true);
      setError(null);
      try {
        const base = mode === "preview" ? transcriptionPdfUrl(stem) : transcriptionSharePdfUrl(stem);
        const response = await fetch(`${base}?t=${refreshKey}`, { signal: controller.signal });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `PDF request failed: ${response.status}`);
        }
        const blob = await response.blob();
        if (!active) return;
        const url = URL.createObjectURL(blob);
        setBlobUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      } catch (err) {
        if (!active || controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : "Failed to load PDF");
        setBlobUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return null;
        });
      } finally {
        if (active) setLoading(false);
      }
    }

    void loadPdf();

    return () => {
      active = false;
      controller.abort();
      setBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
    };
  }, [stem, mode, refreshKey]);

  const handleDownload = async () => {
    try {
      const blob = await fetchTranscriptionPdfBlob(stem, mode);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${stem}_transcription.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    }
  };

  return (
    <aside
      className="flex h-full min-h-0 min-w-0 flex-col bg-gray-50"
      aria-label="Transcription PDF preview"
    >
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-gray-200 bg-white px-3 py-2">
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-gray-900">Transcription PDF</h2>
          <p className="truncate text-[11px] text-gray-500">
            Spatial layout — paired text at segment positions on a blank page
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <div className="flex rounded-md border border-gray-200 bg-gray-50 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => onSwitchMode("preview")}
              className={`rounded px-2 py-1 ${
                mode === "preview" ? "bg-white text-violet-900 shadow-sm" : "text-gray-600 hover:text-gray-900"
              }`}
              title="Live preview from current annotation"
            >
              Preview
            </button>
            <button
              type="button"
              onClick={() => onSwitchMode("share")}
              disabled={!locked}
              className={`rounded px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40 ${
                mode === "share" ? "bg-white text-violet-900 shadow-sm" : "text-gray-600 hover:text-gray-900"
              }`}
              title={locked ? "Frozen PDF from lock time" : "Share PDF available when page is locked"}
            >
              Share
            </button>
          </div>
          <button
            type="button"
            onClick={() => void handleDownload()}
            className="rounded px-2 py-1 text-xs text-violet-800 hover:bg-violet-50"
            title="Download PDF"
          >
            Download
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded px-2 py-1 text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800"
          >
            Close
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-2">
        {mode === "preview" ? (
          <p className="mb-2 shrink-0 px-1 text-[11px] text-gray-500">
            Live draft — updates when annotation is saved.
          </p>
        ) : (
          <p className="mb-2 shrink-0 px-1 text-[11px] text-gray-500">
            Frozen share copy from when the page was locked.
          </p>
        )}

        {loading && (
          <div className="flex flex-1 items-center justify-center text-sm text-gray-500">Loading PDF…</div>
        )}

        {error && !loading && (
          <div className="flex flex-1 items-center justify-center px-4 text-center text-sm text-red-700">
            {error}
          </div>
        )}

        {blobUrl && !loading && !error && (
          <object
            key={blobUrl}
            data={blobUrl}
            type="application/pdf"
            title="Transcription PDF preview"
            className="min-h-0 flex-1 rounded border border-gray-200 bg-white shadow-sm"
          >
            <p className="p-4 text-center text-sm text-gray-500">
              Your browser cannot display PDFs inline. Use Download instead.
            </p>
          </object>
        )}
      </div>
    </aside>
  );
}
