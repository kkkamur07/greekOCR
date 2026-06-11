"use client";

import type { ReactNode } from "react";

interface MenuSectionProps {
  label: string;
  children: ReactNode;
}

export function MenuSection({ label, children }: MenuSectionProps) {
  return (
    <div className="py-1">
      <p className="px-2 pb-1 text-[10px] font-medium uppercase tracking-wide text-gray-400">{label}</p>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

interface MenuRowProps {
  icon: ReactNode;
  label: string;
  hint?: string;
  active?: boolean;
  disabled?: boolean;
  destructive?: boolean;
  prominent?: boolean;
  onClick: () => void;
}

export function MenuRow({
  icon,
  label,
  hint,
  active,
  disabled,
  destructive,
  prominent,
  onClick,
}: MenuRowProps) {
  const wellClass = prominent
    ? "bg-gray-900 text-white"
    : destructive
      ? "bg-red-50 text-red-600"
      : active
        ? "bg-indigo-100 text-indigo-700"
        : "bg-gray-100 text-gray-600";

  const rowClass = prominent
    ? "hover:bg-gray-50"
    : destructive
      ? "hover:bg-red-50"
      : active
        ? "bg-indigo-50/60"
        : "hover:bg-gray-50";

  const labelClass = prominent
    ? "font-medium text-gray-900"
    : destructive
      ? "text-red-800"
      : active
        ? "font-medium text-gray-900"
        : "text-gray-900";

  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onClick}
      className={`flex w-full items-center gap-2.5 rounded-md px-2 py-2 text-left disabled:cursor-not-allowed disabled:opacity-40 ${rowClass}`}
    >
      <span
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${wellClass}`}
        aria-hidden
      >
        {icon}
      </span>
      <span className="min-w-0">
        <span className={`block text-sm ${labelClass}`}>{label}</span>
        {hint && <span className="block text-[11px] text-gray-500">{hint}</span>}
      </span>
    </button>
  );
}
