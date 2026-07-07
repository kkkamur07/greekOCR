import { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { ApiError } from '../../api/errors';
import { toast } from '../ui/toast';

type PublicDocumentDownloadsProps = {
  projectId: string;
  documentId: string;
  partId: string;
  partIndex: number;
  pdfPreviewOpen: boolean;
  onPdfPreviewOpenChange: (open: boolean) => void;
};

function downloadFilename(partIndex: number, extension: string): string {
  return `page-${partIndex}.${extension}`;
}

export function PublicDocumentDownloads({
  projectId,
  documentId,
  partId,
  partIndex,
  pdfPreviewOpen,
  onPdfPreviewOpenChange,
}: PublicDocumentDownloadsProps) {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [loadingPdf, setLoadingPdf] = useState(false);
  const [downloading, setDownloading] = useState<'pdf' | 'xml' | null>(null);

  useEffect(() => {
    if (!pdfPreviewOpen) {
      setPdfUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return null;
      });
      return;
    }

    let active = true;
    setLoadingPdf(true);
    void (async () => {
      try {
        const blob = await api.getPublicTranscriptionPdf(projectId, documentId, partId);
        if (!active) return;
        const url = URL.createObjectURL(blob);
        setPdfUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return url;
        });
      } catch (err) {
        if (!active) return;
        const message = err instanceof ApiError ? err.message : 'Failed to load transcription PDF';
        toast.error(message);
        onPdfPreviewOpenChange(false);
      } finally {
        if (active) setLoadingPdf(false);
      }
    })();

    return () => {
      active = false;
    };
  }, [projectId, documentId, partId, pdfPreviewOpen, onPdfPreviewOpenChange]);

  async function handleDownloadPdf() {
    setDownloading('pdf');
    try {
      const blob = await api.getPublicTranscriptionPdf(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement('a');
      anchor.href = url;
      anchor.download = downloadFilename(partIndex, 'pdf');
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to download PDF';
      toast.error(message);
    } finally {
      setDownloading(null);
    }
  }

  async function handleDownloadXml() {
    setDownloading('xml');
    try {
      const blob = await api.getPublicPageXml(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement('a');
      anchor.href = url;
      anchor.download = downloadFilename(partIndex, 'xml');
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to download PAGE XML';
      toast.error(message);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="pub-downloads">
      <div className="pub-downloads__actions">
        <button
          type="button"
          className={`btn btn-outline btn-sm${pdfPreviewOpen ? ' btn--on' : ''}`}
          aria-pressed={pdfPreviewOpen}
          onClick={() => onPdfPreviewOpenChange(!pdfPreviewOpen)}
        >
          {pdfPreviewOpen ? 'Hide PDF' : 'Preview PDF'}
        </button>
        <button
          type="button"
          className="btn btn-outline btn-sm"
          disabled={downloading !== null}
          onClick={() => void handleDownloadPdf()}
        >
          {downloading === 'pdf' ? 'Downloading…' : 'Download PDF'}
        </button>
        <button
          type="button"
          className="btn btn-outline btn-sm"
          disabled={downloading !== null}
          onClick={() => void handleDownloadXml()}
        >
          {downloading === 'xml' ? 'Downloading…' : 'Download PAGE XML'}
        </button>
      </div>

      {pdfPreviewOpen && (
        <div className="pub-pdf-preview" aria-label="Transcription PDF preview">
          {loadingPdf && <p className="text-sm text-muted">Loading PDF…</p>}
          {!loadingPdf && pdfUrl && (
            <object
              data={pdfUrl}
              type="application/pdf"
              title="Transcription PDF preview"
              className="pub-pdf-preview__frame"
            >
              <p className="text-sm text-muted">
                Your browser cannot display PDFs inline. Use Download PDF instead.
              </p>
            </object>
          )}
        </div>
      )}
    </div>
  );
}
