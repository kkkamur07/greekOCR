/** Lightweight shell shown while AuthProvider restores the session cookie. */
export function SessionRestoring({
  label = "Restoring your session…",
}: {
  label?: string;
}) {
  return (
    <div className="page session-restoring" aria-busy="true" aria-live="polite">
      <nav className="topnav" aria-label="Main navigation">
        <div className="topnav-logo" aria-hidden="true">
          <img src="/nomos.svg" alt="" />
          <span>nomicous</span>
        </div>
      </nav>
      <main className="content-wrap session-restoring__main">
        <div className="session-restoring__panel" role="status">
          <span className="session-restoring__spinner" aria-hidden="true" />
          <p className="session-restoring__label">{label}</p>
        </div>
      </main>
    </div>
  );
}
