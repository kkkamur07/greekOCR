import type { ReactNode } from 'react';
import Link from 'next/link';

type AuthLayoutProps = {
  headline: string;
  tagline: string;
  children: ReactNode;
};

export function AuthLayout({ headline, tagline, children }: AuthLayoutProps) {
  return (
    <div className="auth-page">
      <aside className="auth-brand" aria-label="About nomicous">
        <Link href="/" className="auth-brand-logo" aria-label="nomicous home">
          <img src="/nomos.svg" alt="" />
          <span>nomicous</span>
        </Link>
        <div className="auth-brand-body">
          <h2>{headline}</h2>
          <p>{tagline}</p>
        </div>
        <div className="auth-brand-footer">
          <p>© nomicous 2026</p>
        </div>
      </aside>
      <main className="auth-panel">{children}</main>
    </div>
  );
}

export function AuthFormWrap({ children }: { children: ReactNode }) {
  return (
    <div className="auth-form-wrap">
      <Link href="/" className="mobile-logo auth-brand-logo" aria-label="nomicous home">
        <img src="/nomos.svg" alt="" />
        <span>nomicous</span>
      </Link>
      {children}
    </div>
  );
}
