import type { ReactNode } from 'react';
import { Alert, Space, Spin, Typography } from 'antd';

type PageEditorShellProps = {
  loading: boolean;
  unavailableDescription: string | null;
  toolbar: ReactNode;
  statusAlerts?: ReactNode;
  showStatusAlerts?: boolean;
  children: ReactNode;
};

export function PageEditorShell({
  loading,
  unavailableDescription,
  toolbar,
  statusAlerts,
  showStatusAlerts = false,
  children,
}: PageEditorShellProps) {
  if (loading) {
    return (
      <div style={{ padding: 24 }}>
        <Space>
          <Spin />
          <Typography.Text>Loading page...</Typography.Text>
        </Space>
      </div>
    );
  }

  if (unavailableDescription) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          type="warning"
          showIcon
          message="Page unavailable"
          description={unavailableDescription}
        />
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        flexDirection: 'column',
        overflow: 'hidden',
        background: '#fff',
      }}
    >
      {toolbar}

      {showStatusAlerts && statusAlerts && (
        <div style={{ flexShrink: 0, borderBottom: '1px solid #e5e7eb', padding: 8 }}>
          {statusAlerts}
        </div>
      )}

      <main style={{ display: 'flex', minHeight: 0, flex: 1, flexDirection: 'column' }}>
        {children}
      </main>
    </div>
  );
}
