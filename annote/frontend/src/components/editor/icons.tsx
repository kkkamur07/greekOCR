interface IconProps {
  className?: string;
}

export function PanIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M8 2v12M2 8h12" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M5 5l3-3 3 3M5 11l3 3 3-3" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function SelectIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M3 2l2.5 9.5L7 8l4.5 1.5L3 2z" />
    </svg>
  );
}

export function RectIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <rect x="3" y="4" width="10" height="8" rx="0.5" strokeWidth="1.5" />
    </svg>
  );
}

export function PolyIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M8 2.5L13 6v5l-5 2.5L3 11V6l5-3.5z" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

export function EditIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <circle cx="4" cy="4" r="1.25" strokeWidth="1.25" />
      <circle cx="12" cy="4" r="1.25" strokeWidth="1.25" />
      <circle cx="12" cy="12" r="1.25" strokeWidth="1.25" />
      <circle cx="4" cy="12" r="1.25" strokeWidth="1.25" />
      <path d="M5 4h6M12 5v6M11 12H5M4 11V5" strokeWidth="1" strokeDasharray="1.5 1.5" />
    </svg>
  );
}

export function LinesIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M3 4.5h10M3 8h10M3 11.5h7" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function LockIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <rect x="3.5" y="7" width="9" height="6.5" rx="1" strokeWidth="1.5" />
      <path d="M5.5 7V5a2.5 2.5 0 015 0v2" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function UnlockIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <rect x="3.5" y="7" width="9" height="6.5" rx="1" strokeWidth="1.5" />
      <path d="M5.5 7V5a2.5 2.5 0 015 0" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function ExportIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M8 2.5v8M5.5 7 8 9.5 10.5 7" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 11.5v1.5a1 1 0 001 1h8a1 1 0 001-1v-1.5" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function PdfIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M4.5 2.5h4.5L12 5v8.5a1 1 0 01-1 1H4.5a1 1 0 01-1-1v-10a1 1 0 011-1z" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M9 2.5V5.5H12" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M5.5 8.5h5M5.5 10.5h3.5" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

export function ShareIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <circle cx="12" cy="4" r="1.75" strokeWidth="1.25" />
      <circle cx="4" cy="8" r="1.75" strokeWidth="1.25" />
      <circle cx="12" cy="12" r="1.75" strokeWidth="1.25" />
      <path d="M5.6 7.1l4.8-2.2M5.6 8.9l4.8 2.2" strokeWidth="1.25" />
    </svg>
  );
}

export function CeilingIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M2.5 6.5c2.5-2 8.5-2 11 0" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="2 2" />
      <path d="M3 10.5h10" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function TrashIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M3.5 4.5h9M6 4.5V3.5h4v1" strokeWidth="1.25" strokeLinecap="round" />
      <path d="M5 4.5l.5 8h5l.5-8" strokeWidth="1.25" strokeLinejoin="round" />
    </svg>
  );
}

export function HistoryIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M8 3.5a4.5 4.5 0 104.5 4.5" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M8 2v2l1.25 1.25M8 8.5V11" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function BinarizeIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <circle cx="8" cy="8" r="5" strokeWidth="1.5" />
      <path d="M8 3v10" strokeWidth="1.5" />
    </svg>
  );
}

export function SegmentIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M3 5h10M3 8h10M3 11h7" strokeWidth="1.5" strokeLinecap="round" />
      <rect x="2" y="3" width="12" height="10" rx="1" strokeWidth="1.25" />
    </svg>
  );
}

export function OcrIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M3.5 11.5V5l4.5-2.5L12.5 5v6.5" strokeWidth="1.25" strokeLinejoin="round" />
      <path d="M6 8.5h4M6 10h2.5" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

export function MoreIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <circle cx="4" cy="8" r="1.25" />
      <circle cx="8" cy="8" r="1.25" />
      <circle cx="12" cy="8" r="1.25" />
    </svg>
  );
}

export function WorkflowIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden>
      <path d="M3 4h4v3H3zM9 4h4v3H9zM3 9h4v3H3z" strokeWidth="1.25" strokeLinejoin="round" />
      <path d="M7 5.5h2M5.5 7v2" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}
