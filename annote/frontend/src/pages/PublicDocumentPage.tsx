import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Button, Layout, Result, Space, Tabs, Typography } from 'antd';
import { EditOutlined } from '@ant-design/icons';
import {
  api,
  publicPartMediaUrl,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
  type PublicTranscriptionLayerResponse,
} from '../api/client';
import { ApiError } from '../api/errors';
import { getAccessToken } from '../auth/storage';
import ImageCanvas from '../components/ImageCanvas/ImageCanvas';
import TranscriptionPanel from '../components/TrascriptionPanel/TranscriptionPanel';
import { WorkflowBadge } from '../components/WorkflowBadge';
import type { Region, Transcription } from '../types';

const { Header, Content } = Layout;

export function PublicDocumentPage() {
  const { projectId, documentId } = useParams<{ projectId: string; documentId: string }>();
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(null);
  const [layout, setLayout] = useState<PublicLayoutResponse | null>(null);
  const [layers, setLayers] = useState<PublicTranscriptionLayerResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activePartId, setActivePartId] = useState<string | null>(null);
  const [selectedRegionId, setSelectedRegionId] = useState<number | null>(null);

  const isLoggedIn = !!getAccessToken();

  useEffect(() => {
    if (!projectId || !documentId) return;

    let cancelled = false;
    (async () => {
      setLoading(true);
      setNotFound(false);
      setErrorMessage(null);
      try {
        const [doc, layoutRes, layerList] = await Promise.all([
          api.getPublicDocument(projectId, documentId),
          api.getPublicLayout(projectId, documentId),
          api.listPublicTranscriptions(projectId, documentId),
        ]);
        if (cancelled) return;
        setDocument(doc);
        setLayout(layoutRes);
        setLayers(layerList);
        const sorted = [...(doc.parts ?? [])].sort((a, b) => a.order - b.order);
        setActivePartId(sorted[0]?.id ?? null);
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true);
          return;
        }
        setErrorMessage(err instanceof ApiError ? err.message : 'Failed to load document');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [projectId, documentId]);

  const parts = useMemo(
    () => [...(document?.parts ?? [])].sort((a, b) => a.order - b.order),
    [document],
  );

  const activePart = parts.find((p) => p.id === activePartId) ?? parts[0] ?? null;

  const regions: Region[] = [];
  const transcriptions: Transcription[] = [];

  const imageUrl = activePart ? publicPartMediaUrl(activePart.id) : null;
  const imageDimensions = {
    width: activePart?.width ?? 0,
    height: activePart?.height ?? 0,
  };

  if (notFound) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <Content style={{ padding: 48, maxWidth: 720, margin: '0 auto' }}>
          <Result
            status="404"
            title="Document not available"
            subTitle="This document is not published or does not exist. Only published documents can be viewed without signing in."
          />
        </Content>
      </Layout>
    );
  }

  if (errorMessage) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <Content style={{ padding: 48, maxWidth: 720, margin: '0 auto' }}>
          <Result status="error" title="Could not load document" subTitle={errorMessage} />
        </Content>
      </Layout>
    );
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '0 24px',
          background: '#001529',
        }}
      >
        <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
          <Link to="/" style={{ color: 'inherit' }}>
            greekOCR
          </Link>
        </Typography.Title>
        <Typography.Text style={{ color: 'rgba(255,255,255,0.65)' }}>Public view</Typography.Text>
        <div style={{ flex: 1 }} />
        {isLoggedIn && projectId && documentId && (
          <Link to={`/projects/${projectId}/documents/${documentId}`}>
            <Button type="primary" icon={<EditOutlined />}>
              Open in editor
            </Button>
          </Link>
        )}
        {!isLoggedIn && (
          <Link to="/login">
            <Button style={{ color: '#fff' }} type="link">
              Sign in
            </Button>
          </Link>
        )}
      </Header>

      <Content style={{ padding: 24, background: '#f0f2f5' }}>
        <Space direction="vertical" size="middle" style={{ width: '100%', marginBottom: 16 }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            {document?.name ?? 'Document'}
          </Typography.Title>
          {document && <WorkflowBadge workflow={document.workflow} />}
          {layers.length > 0 && (
            <Typography.Text type="secondary">
              Transcription layers: {layers.map((l) => l.name).join(', ')}
            </Typography.Text>
          )}
          {layout && (layout.blocks?.length ?? 0) + (layout.lines?.length ?? 0) > 0 && (
            <Typography.Text type="secondary">
              Layout: {layout.blocks?.length ?? 0} blocks, {layout.lines?.length ?? 0} lines
            </Typography.Text>
          )}
        </Space>

        {parts.length > 1 && (
          <Tabs
            activeKey={activePart?.id}
            onChange={setActivePartId}
            items={parts.map((part, index) => ({
              key: part.id,
              label: `Part ${index + 1}`,
            }))}
            style={{ marginBottom: 16 }}
          />
        )}

        <div style={{ display: 'flex', gap: 24, minHeight: 'calc(100vh - 220px)' }}>
          <div
            style={{
              flex: '0 0 65%',
              background: '#fff',
              borderRadius: 8,
              overflow: 'hidden',
              opacity: loading ? 0.6 : 1,
            }}
          >
            <ImageCanvas
              readOnly
              imageUrl={imageUrl}
              imageDimensions={imageDimensions}
              regions={regions}
              selectedRegionId={selectedRegionId}
              onSelectRegion={setSelectedRegionId}
              onAddRegion={() => {}}
              onUpdateRegion={() => {}}
              onDeleteRegion={() => {}}
              onTranscribeRegion={() => {}}
            />
          </div>
          <div style={{ flex: '0 0 35%', background: '#fff', borderRadius: 8, overflow: 'hidden' }}>
            <TranscriptionPanel
              readOnly
              regions={regions}
              transcriptions={transcriptions}
              selectedRegionId={selectedRegionId}
              onSelectRegion={setSelectedRegionId}
              onUpdateTranscription={() => {}}
              onDeleteRegion={() => {}}
            />
          </div>
        </div>

        {!loading && parts.length === 0 && (
          <Typography.Text type="secondary">
            This published document has no page images yet.
          </Typography.Text>
        )}
      </Content>
    </Layout>
  );
}
