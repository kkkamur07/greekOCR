import { useEffect, useId, type FormEvent, type ReactNode } from 'react';

type FormModalProps = {
  open: boolean;
  title: string;
  onClose: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  submitLabel: string;
  loading?: boolean;
  children: ReactNode;
};

export function FormModal({
  open,
  title,
  onClose,
  onSubmit,
  submitLabel,
  loading = false,
  children,
}: FormModalProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="modal-overlay"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="modal-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id={titleId}>{title}</h2>
        <form onSubmit={onSubmit}>
          {children}
          <button type="submit" className="btn btn-primary btn-block mt-4" disabled={loading}>
            {loading ? 'Saving…' : submitLabel}
          </button>
        </form>
      </div>
    </div>
  );
}
