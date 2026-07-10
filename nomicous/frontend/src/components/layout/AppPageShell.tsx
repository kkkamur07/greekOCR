import type { ReactNode } from "react";
import { PageHeader } from "./PageHeader";
import { TopNav, type BreadcrumbItem } from "./TopNav";

type AppPageShellProps = {
  breadcrumb?: BreadcrumbItem[];
  currentLabel?: string;
  username?: string | null;
  navActions?: ReactNode;
  title: string;
  subtitle?: string;
  titleExtra?: ReactNode;
  titleEditable?: boolean;
  titlePanelOpen?: boolean;
  onTitlePanelToggle?: () => void;
  titlePanel?: ReactNode;
  titlePanelLabel?: string;
  headerActions?: ReactNode;
  children: ReactNode;
};

export function AppPageShell({
  breadcrumb,
  currentLabel,
  username,
  navActions,
  title,
  subtitle,
  titleExtra,
  titleEditable,
  titlePanelOpen,
  onTitlePanelToggle,
  titlePanel,
  titlePanelLabel,
  headerActions,
  children,
}: AppPageShellProps) {
  return (
    <div className="page">
      <TopNav
        breadcrumb={breadcrumb}
        currentLabel={currentLabel}
        username={username}
        actions={navActions}
      />
      <PageHeader
        title={title}
        subtitle={subtitle}
        titleExtra={titleExtra}
        titleEditable={titleEditable}
        titlePanelOpen={titlePanelOpen}
        onTitlePanelToggle={onTitlePanelToggle}
        titlePanel={titlePanel}
        titlePanelLabel={titlePanelLabel}
        actions={headerActions}
      />
      <main className="content-wrap">{children}</main>
    </div>
  );
}
