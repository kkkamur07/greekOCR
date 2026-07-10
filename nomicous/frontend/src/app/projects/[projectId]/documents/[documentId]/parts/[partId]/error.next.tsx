'use client';

type ErrorBoundaryProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

export default function PartEditorError({ error, reset }: ErrorBoundaryProps) {
  return (
    <section role="alert" aria-live="assertive">
      <h2>Unable to open the page editor</h2>
      <p>The editor could not load. Your work has not been changed.</p>
      {error.digest ? <p>Reference: {error.digest}</p> : null}
      <button type="button" onClick={reset}>
        Try again
      </button>
    </section>
  );
}
