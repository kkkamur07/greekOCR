import { Layout, Button, Typography } from 'antd';
import { Link, useNavigate } from 'react-router-dom';
import { clearAccessToken } from '../auth/storage';

const { Header, Content } = Layout;

export function AppLayout({
  title,
  children,
  extra,
}: {
  title: string;
  children: React.ReactNode;
  extra?: React.ReactNode;
}) {
  const navigate = useNavigate();

  const handleLogout = () => {
    clearAccessToken();
    navigate('/login');
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '0 24px',
        }}
      >
        <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
          <Link to="/projects" style={{ color: 'inherit' }}>
            greekOCR
          </Link>
        </Typography.Title>
        <div style={{ flex: 1 }} />
        {extra}
        <Button type="link" onClick={handleLogout} style={{ color: '#fff' }}>
          Log out
        </Button>
      </Header>
      <Content style={{ padding: 24, maxWidth: 1100, margin: '0 auto', width: '100%' }}>
        <Typography.Title level={3}>{title}</Typography.Title>
        {children}
      </Content>
    </Layout>
  );
}
