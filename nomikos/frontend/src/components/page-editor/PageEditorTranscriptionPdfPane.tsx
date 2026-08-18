import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import {
  CloseIcon,
  DownloadIcon,
  IconButton,
  RefreshIcon,
} from "./EditorIcons";

type PageEditorTranscriptionPdfPaneProps = {
  projectId: string;
  documentId: string;
  partId: string;
  downloadFilename: string;
  refreshKey: number;
  onClose: () => void;
  onRefresh: () => void;
};

function transcriptionPdfErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  return error instanceof Error
    ? error.message
    : "Failed to load transcription PDF";
}

export function PageEditorTranscriptionPdfPane({
  projectId,
  documentId,
  partId,
  downloadFilename,
  refreshKey,
  onClose,
  onRefresh,
}: PageEditorTranscriptionPdfPaneProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadPdf() {
      setLoading(true);
      setError(null);
      try {
        const blob = await api.generateTranscriptionPdf(
          projectId,
          documentId,
          partId,
        );
        if (!active) return;
        const url = URL.createObjectURL(blob);
        setBlobUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return url;
        });
      } catch (err) {
        if (!active) return;
        setError(transcriptionPdfErrorMessage(err));
        setBlobUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return null;
        });
      } finally {
        if (active) setLoading(false);
      }
    }

    void loadPdf();

    return () => {
      active = false;
      setBlobUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return null;
      });
    };
  }, [projectId, documentId, partId, refreshKey]);

  async function handleDownload() {
    setDownloading(true);
    setError(null);
    try {
      const blob = await api.generateTranscriptionPdf(
        projectId,
        documentId,
        partId,
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(transcriptionPdfErrorMessage(err));
    } finally {
      setDownloading(false);
    }
  }

  return (
    <aside className="pe-pdf-pane" aria-label="Transcription PDF preview">
      <div className="pe-pdf-pane__header">
        <span>Transcription PDF</span>
        <div className="pe-pdf-pane__actions">
          <IconButton
            label="Refresh PDF"
            onClick={onRefresh}
            disabled={loading}
          >
            <RefreshIcon />
          </IconButton>
          <IconButton
            label="Download PDF"
            onClick={() => void handleDownload()}
            disabled={downloading}
          >
            <DownloadIcon />
          </IconButton>
          <IconButton label="Close PDF pane" onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </div>
      </div>

      <div
        className="pe-pdf-pane__body"
        style={{ padding: 8, alignItems: "stretch" }}
      >
        {loading && <span>Loading PDF…</span>}
        {error && !loading && (
          <span style={{ color: "#991b1b", padding: 16, textAlign: "center" }}>
            {error}
          </span>
        )}
        {blobUrl && !loading && !error && (
          <>
            <p
              style={{
                flexShrink: 0,
                margin: 0,
                padding: "0 4px 6px",
                color: "var(--text-3)",
              }}
            >
              If the preview stays blank,{" "}
              <a
                href={blobUrl}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "inherit" }}
              >
                open the PDF in a new tab
              </a>{" "}
              or use Download.
            </p>
            <iframe
              key={blobUrl}
              src={blobUrl}
              title="Transcription PDF preview"
              style={{
                minHeight: 0,
                flex: 1,
                border: "1px solid var(--border)",
                borderRadius: 6,
                background: "#fff",
              }}
            />
          </>
        )}
      </div>
    </aside>
  );
}
