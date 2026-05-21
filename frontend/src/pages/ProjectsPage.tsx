import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Button,
  Form,
  Input,
  List,
  Modal,
  Space,
  Tag,
  Typography,
  notification,
} from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import { api, type ProjectResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { AppLayout } from '../components/AppLayout';

function slugify(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 512) || 'project';
}

export function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [userId, setUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const [me, list] = await Promise.all([api.me(), api.listProjects()]);
      setUserId(me.id);
      setProjects(list);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to load projects';
      notification.error({ message: msg });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const owned = projects.filter((p) => p.owner_id === userId);
  const shared = projects.filter((p) => p.owner_id !== userId);

  const handleCreate = async (values: { name: string }) => {
    setCreating(true);
    try {
      await api.createProject({ name: values.name, slug: slugify(values.name) });
      notification.success({ message: 'Project created' });
      setModalOpen(false);
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to create project';
      notification.error({ message: msg });
    } finally {
      setCreating(false);
    }
  };

  const renderList = (items: ProjectResponse[], emptyText: string) => (
    <List
      loading={loading}
      dataSource={items}
      locale={{ emptyText }}
      renderItem={(project) => (
        <List.Item>
          <List.Item.Meta
            title={<Link to={`/projects/${project.id}`}>{project.name}</Link>}
            description={
              <Space>
                <Typography.Text type="secondary">{project.slug}</Typography.Text>
                {project.owner_id !== userId && <Tag>shared</Tag>}
              </Space>
            }
          />
        </List.Item>
      )}
    />
  );

  return (
    <AppLayout
      title="Projects"
      extra={
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
          New project
        </Button>
      }
    >
      <Typography.Title level={5}>Owned</Typography.Title>
      {renderList(owned, 'No owned projects yet')}

      <Typography.Title level={5} style={{ marginTop: 24 }}>
        Shared with me
      </Typography.Title>
      {renderList(shared, 'No shared projects')}

      <Modal
        title="New project"
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
