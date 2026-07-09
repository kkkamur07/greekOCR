import { useEffect, useRef, type ReactNode } from 'react';

type PageHeaderProps = {
  title: string;
  subtitle?: string;
  titleExtra?: ReactNode;
  actions?: ReactNode;
  titleEditable?: boolean;
  titlePanelOpen?: boolean;
  onTitlePanelToggle?: () => void;
  titlePanel?: ReactNode;
  titlePanelLabel?: string;
};

export function PageHeader({
  title,
  subtitle,
  titleExtra,
  actions,
  titleEditable = false,
  titlePanelOpen = false,
  onTitlePanelToggle,
  titlePanel,
  titlePanelLabel = 'Edit details',
}: PageHeaderProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!titlePanelOpen) return;
    function handleClick(event: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) {
        onTitlePanelToggle?.();
      }
    }
    globalThis.document.addEventListener('click', handleClick);
    return () => globalThis.document.removeEventListener('click', handleClick);
  }, [titlePanelOpen, onTitlePanelToggle]);

  return (
    <header className="page-header">
      <div className="page-header__main" ref={panelRef}>
        <div className="page-header__title-row">
          {titleEditable ? (
            <button
              type="button"
              className="page-header__title-btn"
              aria-expanded={titlePanelOpen}
              aria-haspopup="dialog"
              aria-label={`${title}, click to edit`}
              onClick={(event) => {
                event.stopPropagation();
                onTitlePanelToggle?.();
              }}
            >
              <h1>{title}</h1>
            </button>
          ) : (
            <h1>{title}</h1>
          )}
          {titleExtra}
        </div>
        {subtitle && <p className="subtitle">{subtitle}</p>}
        {titlePanelOpen && titlePanel && (
          <div className="entity-panel" role="dialog" aria-label={titlePanelLabel}>
            {titlePanel}
          </div>
        )}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </header>
  );
}
