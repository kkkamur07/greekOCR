import { useCallback, useEffect, useRef } from "react";

type PublicPartTabsProps = {
  parts: { id: string; label: string }[];
  activeId: string | null;
  onChange: (id: string) => void;
  variant?: "default" | "workspace";
};

export function PublicPartTabs({
  parts,
  activeId,
  onChange,
  variant = "default",
}: PublicPartTabsProps) {
  const tabsRef = useRef<(HTMLButtonElement | null)[]>([]);

  const selectTab = useCallback(
    (index: number) => {
      const tab = parts[index];
      if (!tab) return;
      onChange(tab.id);
      parts.forEach((_, i) => {
        const el = tabsRef.current[i];
        if (!el) return;
        const selected = i === index;
        el.setAttribute("aria-selected", String(selected));
        el.tabIndex = selected ? 0 : -1;
      });
    },
    [parts, onChange],
  );

  useEffect(() => {
    const idx = parts.findIndex((p) => p.id === activeId);
    if (idx < 0) return;
    selectTab(idx);
    // No smooth-scroll behavior here on purpose: it is unnecessary motion, and
    // some viewers set prefers-reduced-motion, so a plain jump keeps this a
    // no-op for anyone who wants that. scrollIntoView is also absent in jsdom.
    const el = tabsRef.current[idx];
    if (typeof el?.scrollIntoView === "function") {
      el.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }, [activeId, parts, selectTab]);

  if (parts.length === 0) return null;

  const wrapClass =
    variant === "workspace" ? "pub-workspace__tabs" : "pub-tabs-wrap";

  return (
    <div className={wrapClass}>
      <div className="tabs" role="tablist" aria-label="Document parts">
        {parts.map((part, index) => {
          const selected = part.id === activeId;
          return (
            <button
              key={part.id}
              ref={(el) => {
                tabsRef.current[index] = el;
              }}
              type="button"
              className={`tab${selected ? " active" : ""}`}
              role="tab"
              id={`tab-${part.id}`}
              aria-selected={selected}
              aria-controls={`panel-${part.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => selectTab(index)}
              onKeyDown={(e) => {
                let next = index;
                if (e.key === "ArrowRight") next = (index + 1) % parts.length;
                else if (e.key === "ArrowLeft")
                  next = (index - 1 + parts.length) % parts.length;
                else if (e.key === "Home") next = 0;
                else if (e.key === "End") next = parts.length - 1;
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
      </div>
    </div>
  );
}
