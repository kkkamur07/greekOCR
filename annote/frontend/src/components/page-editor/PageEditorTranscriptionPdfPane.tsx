import { useEffect, useState } from 'react';
import { Button, Space, Typography } from 'antd';
import { api } from '../../api/client';
import { ApiError } from '../../api/errors';

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
  return error instanceof Error ? error.message : 'Failed to load transcription PDF';
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
        const blob = await api.generateTranscriptionPdf(projectId, documentId, partId);
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
      const blob = await api.generateTranscriptionPdf(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
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
    <aside
      aria-label="Transcription PDF preview"
      style={{
        display: 'flex',
        minHeight: 0,
        minWidth: 0,
        width: '50%',
        flexDirection: 'column',
        borderLeft: '1px solid #e5e7eb',
        background: '#f9fafb',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexShrink: 0,
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          borderBottom: '1px solid #e5e7eb',
          background: '#fff',
          padding: '8px 12px',
        }}
      >
        <div style={{ minWidth: 0 }}>
          <Typography.Text strong>Transcription PDF</Typography.Text>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0, fontSize: 12 }}>
            Paired ground truth text at segment positions on a blank page
          </Typography.Paragraph>
        </div>
        <Space size={4} wrap>
          <Button size="small" onClick={onRefresh} disabled={loading}>
            Refresh
          </Button>
          <Button size="small" loading={downloading} onClick={() => void handleDownload()}>
            Download
          </Button>
          <Button size="small" onClick={onClose}>
            Close
          </Button>
        </Space>
      </div>

      <div
        style={{
          display: 'flex',
          minHeight: 0,
          flex: 1,
          flexDirection: 'column',
          overflow: 'hidden',
          padding: 8,
        }}
      >
        <Typography.Paragraph type="secondary" style={{ marginBottom: 8, fontSize: 12 }}>
          Live preview from current pairing and ground truth. Refresh after saving changes.
        </Typography.Paragraph>

        {loading && (
          <div
            style={{
              display: 'flex',
              flex: 1,
              alignItems: 'center',
              justifyContent: 'center',
              color: '#6b7280',
            }}
          >
            Loading PDF…
          </div>
        )}

        {error && !loading && (
          <div
            style={{
              display: 'flex',
              flex: 1,
              alignItems: 'center',
              justifyContent: 'center',
              paddingInline: 16,
              textAlign: 'center',
              color: '#b91c1c',
            }}
          >
            {error}
          </div>
        )}

        {blobUrl && !loading && !error && (
          <object
            key={blobUrl}
            data={blobUrl}
            type="application/pdf"
            title="Transcription PDF preview"
            aria-label="Transcription PDF preview"
            style={{
              minHeight: 0,
              flex: 1,
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              background: '#fff',
            }}
          >
            <Typography.Paragraph type="secondary" style={{ padding: 16, textAlign: 'center' }}>
              Your browser cannot display PDFs inline. Use Download instead.
            </Typography.Paragraph>
          </object>
        )}
      </div>
    </aside>
  );
}
