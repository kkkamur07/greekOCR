import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button, Card, Form, Input, Typography, notification } from 'antd';
import { api, type LoginRequest } from '../api/client';
import { ApiError } from '../api/errors';
import { setAccessToken } from '../auth/storage';

export function LoginPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const onFinish = async (values: LoginRequest) => {
    setLoading(true);
    try {
      const token = await api.login(values);
      setAccessToken(token.access_token);
      notification.success({ message: 'Signed in' });
      navigate('/projects');
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Login failed';
      notification.error({ message: msg });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#f0f2f5',
      }}
    >
      <Card style={{ width: 400 }}>
        <Typography.Title level={3}>Sign in</Typography.Title>
        <Form layout="vertical" onFinish={onFinish}>
          <Form.Item
            name="email"
            label="Email"
            rules={[{ required: true, type: 'email' }]}
          >
            <Input autoComplete="email" />
          </Form.Item>
          <Form.Item name="password" label="Password" rules={[{ required: true }]}>
            <Input.Password autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            Sign in
          </Button>
        </Form>
        <Typography.Paragraph style={{ marginTop: 16, marginBottom: 0 }}>
          No account? <Link to="/register">Register</Link>
        </Typography.Paragraph>
      </Card>
    </div>
  );
}
