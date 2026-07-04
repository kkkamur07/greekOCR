import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  Alert,
  Button,
  Checkbox,
  Form,
  Input,
  List,
  Modal,
  Space,
  Typography,
  notification,
} from 'antd';
import { PlusOutlined, ArrowLeftOutlined } from '@ant-design/icons';
import { api, type DocumentResponse, type ProjectResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { AppLayout } from '../components/AppLayout';
import { WorkflowBadge } from '../components/WorkflowBadge';

export function ProjectDashboardPage() {
  const navigate = useNavigate();
  const { projectId } = useParams<{ projectId: string }>();
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [includeArchived, setIncludeArchived] = useState(false);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const [proj, docs] = await Promise.all([
        api.getProject(projectId),
        api.listDocuments(projectId, includeArchived),
      ]);
      setProject(proj);
      setDocuments(docs);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to load project';
      setProject(null);
      setDocuments([]);
      setError(
        err instanceof ApiError && (err.status === 403 || err.status === 404)
          ? 'This project is not available to your account.'
          : msg,
      );
      notification.error({ message: msg });
    } finally {
      setLoading(false);
    }
  }, [projectId, includeArchived]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleCreate = async (values: { name: string }) => {
    if (!projectId) return;
    setCreating(true);
    try {
      const doc = await api.createDocument(projectId, { name: values.name });
      notification.success({ message: 'Document created' });
      setModalOpen(false);
      navigate(`/projects/${projectId}/documents/${doc.id}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to create document';
      notification.error({ message: msg });
    } finally {
      setCreating(false);
    }
  };

  return (
    <AppLayout
      title={project?.name ?? 'Project'}
      extra={
        <Link to="/projects">
          <Button icon={<ArrowLeftOutlined />}>Projects</Button>
        </Link>
      }
    >
      {error && (
        <Alert
          type="warning"
          showIcon
          message="Project unavailable"
          description={error}
          style={{ marginBottom: 16 }}
        />
      )}

      {project && (
        <Space style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
            New document
          </Button>
          <Checkbox
            checked={includeArchived}
            onChange={(e) => setIncludeArchived(e.target.checked)}
          >
            Show archived
          </Checkbox>
        </Space>
      )}

      {project && (
        <List
          loading={loading}
          dataSource={documents}
          locale={{ emptyText: 'No documents yet' }}
          renderItem={(doc) => (
            <List.Item
              actions={[
                <Link key="open" to={`/projects/${projectId}/documents/${doc.id}`}>
                  Open
                </Link>,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Link to={`/projects/${projectId}/documents/${doc.id}`}>{doc.name}</Link>
                    <WorkflowBadge workflow={doc.workflow} />
                  </Space>
                }
                description={
                  <Typography.Text type="secondary">
                    Updated {new Date(doc.updated_at).toLocaleString()}
                  </Typography.Text>
                }
              />
            </List.Item>
          )}
        />
      )}

      <Modal
        title="New document"
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        footer={null}
        destroyOnHidden
      >
        <Form layout="vertical" onFinish={handleCreate}>
          <Form.Item name="name" label="Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={creating} block>
            Create
          </Button>
        </Form>
      </Modal>
    </AppLayout>
  );
}
