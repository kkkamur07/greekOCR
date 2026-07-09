import type { ReactNode } from 'react';
import { Spin, Typography } from 'antd';
import { PageEditorNavHeader } from './PageEditorNavHeader';

type PageEditorShellProps = {
  loading: boolean;
  unavailableDescription: string | null;
  backHref: string;
  toolbar: ReactNode;
  processingBanner?: ReactNode;
  inferenceBanner?: ReactNode;
  statusAlerts?: ReactNode;
  showStatusAlerts?: boolean;
  children: ReactNode;
};

export function PageEditorShell({
  loading,
  unavailableDescription,
  backHref,
  toolbar,
  processingBanner,
  inferenceBanner,
  statusAlerts,
  showStatusAlerts = false,
  children,
}: PageEditorShellProps) {
  if (loading) {
    return (
      <div className="pe-shell">
        <PageEditorNavHeader backHref={backHref} />
        <div className="pe-shell__message">
          <Spin />
          <Typography.Text>Loading page…</Typography.Text>
        </div>
      </div>
    );
  }

  if (unavailableDescription) {
    return (
      <div className="pe-shell">
        <PageEditorNavHeader backHref={backHref} />
        <div className="pe-shell__message">
          <Typography.Title level={4}>Page unavailable</Typography.Title>
          <Typography.Text type="secondary">{unavailableDescription}</Typography.Text>
        </div>
      </div>
    );
  }

  return (
    <div className="pe-shell">
      {toolbar}

      {inferenceBanner}

      {processingBanner}

      {showStatusAlerts && statusAlerts && (
        <div className="pe-status-bar" role="status" aria-live="polite">
          {statusAlerts}
        </div>
      )}

      <main className="pe-main">{children}</main>
    </div>
  );
}
