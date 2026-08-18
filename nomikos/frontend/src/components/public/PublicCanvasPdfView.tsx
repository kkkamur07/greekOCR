import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { PublicZoomSurface } from "./PublicZoomSurface";

type PublicCanvasPdfViewProps = {
  projectId: string;
  documentId: string;
  partId: string;
  downloadFilename?: string;
};

export function PublicCanvasPdfView({
  projectId,
  documentId,
  partId,
  downloadFilename = "transcription.pdf",
}: PublicCanvasPdfViewProps) {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    void (async () => {
      try {
        const blob = await api.getPublicTranscriptionPdf(
          projectId,
          documentId,
          partId,
        );
        if (!active) return;
        const url = URL.createObjectURL(blob);
        setPdfUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return url;
        });
      } catch (err) {
        if (!active) return;
        setError(
          err instanceof ApiError
            ? err.message
            : "Failed to load transcription PDF",
        );
        setPdfUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return null;
        });
      } finally {
        if (active) setLoading(false);
      }
    })();

    return () => {
      active = false;
      setPdfUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return null;
      });
    };
  }, [projectId, documentId, partId]);

  if (loading) {
    return (
      <p className="pub-pdf-view__status text-sm text-muted">
        Loading transcription PDF…
      </p>
    );
  }

  if (error) {
    return <p className="pub-pdf-view__status text-sm text-muted">{error}</p>;
  }

  if (!pdfUrl) {
    return (
      <p className="pub-pdf-view__status text-sm text-muted">
        No transcription PDF available for this page.
      </p>
    );
  }

  return (
    <div
      className="pub-pdf-view"
      style={{ display: "flex", flexDirection: "column", flex: 1 }}
    >
      <div
        className="pub-pdf-view__actions"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          justifyContent: "center",
          padding: 8,
          flexShrink: 0,
        }}
      >
        <a
          className="btn btn-outline btn-sm"
          href={pdfUrl}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open PDF in new tab
        </a>
        <a
          className="btn btn-outline btn-sm"
          href={pdfUrl}
          download={downloadFilename}
        >
          Download PDF
        </a>
      </div>
      <PublicZoomSurface ariaLabel="Transcription PDF viewer">
        <div className="pub-pdf-view__frame-wrap">
          <iframe
            src={pdfUrl}
            title="Transcription PDF"
            className="pub-pdf-view__frame"
          />
        </div>
      </PublicZoomSurface>
    </div>
  );
}
