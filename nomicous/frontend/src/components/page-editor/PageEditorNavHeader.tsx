import Link from 'next/link';

type PageEditorBackLinkProps = {
  to: string;
  label?: string;
};

export function PageEditorBackLink({ to, label = 'Back' }: PageEditorBackLinkProps) {
  return (
    <Link href={to} className="pe-toolbar__back" aria-label="Back to document">
      <span className="pe-toolbar__back-icon" aria-hidden="true">
        ←
      </span>
      <span className="pe-toolbar__back-label">{label}</span>
    </Link>
  );
}

type PageEditorNavHeaderProps = {
  backHref: string;
};

export function PageEditorNavHeader({ backHref }: PageEditorNavHeaderProps) {
  return (
    <header className="pe-toolbar pe-toolbar--minimal" role="banner">
      <Link href="/projects" className="pe-toolbar__logo" aria-label="nomicous home">
        <img src="/nomos.svg" alt="" />
        <span>nomicous</span>
      </Link>
      <PageEditorBackLink to={backHref} />
    </header>
  );
}
