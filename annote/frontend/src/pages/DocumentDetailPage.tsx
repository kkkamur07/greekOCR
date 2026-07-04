import { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Space,
  Typography,
  Upload,
  notification,
} from 'antd';
import {
  ArrowLeftOutlined,
  ArrowDownOutlined,
  ArrowUpOutlined,
  DeleteOutlined,
  GlobalOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { api, type DocumentWithPartsResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { AppLayout } from '../components/AppLayout';
import { AuthenticatedImage } from '../components/AuthenticatedImage';
import { JobsPanel } from '../components/JobsPanel/JobsPanel';
import { WorkflowBadge } from '../components/WorkflowBadge';

const ENABLE_TEST_JOBS =
  (import.meta.env.VITE_ENABLE_TEST_JOBS as string | undefined) === 'true';

export function DocumentDetailPage() {
  const { projectId, documentId } = useParams<{ projectId: string; documentId: string }>();
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reordering, setReordering] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!projectId || !documentId) return;
    setLoading(true);
    setError(null);
    try {
      const doc = await api.getDocument(projectId, documentId);
      setDocument(doc);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to load document';
      setDocument(null);
      setError(
        err instanceof ApiError && (err.status === 403 || err.status === 404)
          ? 'This document is not available to your account.'
          : msg,
      );
      notification.error({ message: msg });
    } finally {
      setLoading(false);
    }
  }, [projectId, documentId]);

  useEffect(() => {
    void load();
  }, [load]);

  const parts = [...(document?.parts ?? [])].sort((a, b) => a.order - b.order);

  const handleUpload = async (file: File) => {
    if (!projectId || !documentId) return false;
    setUploading(true);
    try {
      await api.uploadPart(projectId, documentId, file);
      notification.success({ message: 'Part uploaded' });
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Upload failed';
      notification.error({ message: msg });
    } finally {
      setUploading(false);
    }
    return false;
  };

  const persistOrder = async (partIds: string[]) => {
    if (!projectId || !documentId) return;
    setReordering(true);
    try {
      await api.reorderParts(projectId, documentId, { part_ids: partIds });
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Reorder failed';
      notification.error({ message: msg });
    } finally {
      setReordering(false);
    }
  };

  const movePart = (index: number, direction: -1 | 1) => {
    const next = index + direction;
    if (next < 0 || next >= parts.length) return;
    const ids = parts.map((p) => p.id);
    [ids[index], ids[next]] = [ids[next], ids[index]];
    void persistOrder(ids);
  };

  const handleDelete = async (partId: string) => {
    if (!projectId || !documentId) return;
    try {
      await api.deletePart(projectId, documentId, partId);
      notification.success({ message: 'Part removed' });
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Delete failed';
      notification.error({ message: msg });
    }
  };

  return (
    <AppLayout
      title={document?.name ?? 'Document'}
      extra={
        projectId ? (
          <Link to={`/projects/${projectId}`}>
            <Button icon={<ArrowLeftOutlined />}>Back to project</Button>
          </Link>
        ) : null
      }
    >
      {document && (
        <Space style={{ marginBottom: 16 }} wrap>
          <WorkflowBadge workflow={document.workflow} />
          {document.workflow === 'published' && projectId && documentId && (
            <Link
              to={`/public/projects/${projectId}/documents/${documentId}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button icon={<GlobalOutlined />} size="small">
                View public page
              </Button>
            </Link>
          )}
        </Space>
      )}

      {error && (
        <Alert
          type="warning"
          showIcon
          message="Document unavailable"
          description={error}
          style={{ marginBottom: 16 }}
        />
      )}

      {document && <JobsPanel enableTestJobs={ENABLE_TEST_JOBS} />}

      {document && (
        <Upload
          accept="image/*"
          showUploadList={false}
          beforeUpload={handleUpload}
          disabled={uploading || loading}
        >
          <Button icon={<UploadOutlined />} loading={uploading}>
            Upload part image
          </Button>
        </Upload>
      )}

      {document && (
        <Typography.Paragraph type="secondary" style={{ marginTop: 8 }}>
          Document parts are Page images in reading order. Open a Page to enter the
          annote editor workflow.
        </Typography.Paragraph>
      )}

      <Space direction="vertical" size="middle" style={{ width: '100%', marginTop: 16 }}>
        {parts.map((part, index) => (
          <Card
            key={part.id}
            loading={loading || reordering}
            title={`Part ${index + 1}`}
            extra={
              <Space>
                <Button
                  size="small"
                  icon={<ArrowUpOutlined />}
                  disabled={index === 0 || reordering}
                  onClick={() => movePart(index, -1)}
                />
                <Button
                  size="small"
                  icon={<ArrowDownOutlined />}
                  disabled={index === parts.length - 1 || reordering}
                  onClick={() => movePart(index, 1)}
                />
                <Button
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => void handleDelete(part.id)}
                />
                <Link to={`/projects/${projectId}/documents/${documentId}/parts/${part.id}`}>
                  <Button size="small" type="primary">
                    Open page
                  </Button>
                </Link>
              </Space>
            }
          >
            <AuthenticatedImage src={part.image_url} alt={`Part ${index + 1}`} width={200} />
            <Typography.Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
              {part.width && part.height
                ? `${part.width}×${part.height}px`
                : 'Dimensions pending'}
            </Typography.Text>
          </Card>
        ))}
        {!loading && document && parts.length === 0 && (
          <Typography.Text type="secondary">No parts yet — upload an image.</Typography.Text>
        )}
      </Space>
    </AppLayout>
  );
}
