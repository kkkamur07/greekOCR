import { useCallback, useEffect, useRef, useState } from 'react';

type PublicPartTabsProps = {
  parts: { id: string; label: string }[];
  activeId: string | null;
  onChange: (id: string) => void;
  maxVisible?: number;
  variant?: 'default' | 'workspace';
};

export function PublicPartTabs({
  parts,
  activeId,
  onChange,
  maxVisible = 6,
  variant = 'default',
}: PublicPartTabsProps) {
  const tabsRef = useRef<(HTMLButtonElement | null)[]>([]);

  const visible = parts.slice(0, maxVisible);
  const overflow = parts.length - visible.length;

  const selectTab = useCallback(
    (index: number) => {
      const tab = visible[index];
      if (!tab) return;
      onChange(tab.id);
      visible.forEach((_, i) => {
        const el = tabsRef.current[i];
        if (!el) return;
        const selected = i === index;
        el.setAttribute('aria-selected', String(selected));
        el.tabIndex = selected ? 0 : -1;
      });
    },
    [visible, onChange],
  );

  useEffect(() => {
    const idx = visible.findIndex((p) => p.id === activeId);
    if (idx >= 0) selectTab(idx);
  }, [activeId, visible, selectTab]);

  if (parts.length === 0) return null;

  const wrapClass =
    variant === 'workspace' ? 'pub-workspace__tabs' : 'pub-tabs-wrap';

  return (
    <div className={wrapClass}>
      <div className="tabs" role="tablist" aria-label="Document parts">
        {visible.map((part, index) => {
          const selected = part.id === activeId;
          return (
            <button
              key={part.id}
              ref={(el) => {
                tabsRef.current[index] = el;
              }}
              type="button"
              className={`tab${selected ? ' active' : ''}`}
              role="tab"
              id={`tab-${part.id}`}
              aria-selected={selected}
              aria-controls={`panel-${part.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => selectTab(index)}
              onKeyDown={(e) => {
                let next = index;
                if (e.key === 'ArrowRight') next = (index + 1) % visible.length;
                else if (e.key === 'ArrowLeft') next = (index - 1 + visible.length) % visible.length;
                else if (e.key === 'Home') next = 0;
                else if (e.key === 'End') next = visible.length - 1;
                else return;
                e.preventDefault();
                selectTab(next);
                tabsRef.current[next]?.focus();
              }}
            >
              {part.label}
            </button>
          );
        })}
        {overflow > 0 && (
          <span className="tab-more" aria-hidden="true">
            + {overflow}
          </span>
        )}
      </div>
    </div>
  );
}
