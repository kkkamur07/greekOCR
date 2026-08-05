/**
 * One local run, handed to the caller when the run begins.
 *
 * Every way a local run can be aborted owns its own signal, because the three
 * causes need three different outcomes and an `AbortError` alone cannot tell
 * them apart:
 *
 * - the jobs-panel signal passed to `trackLocalTask` - the user cancelled, so
 *   the run stops and nothing is sent to the cloud;
 * - `cloudSwitchSignal` - the page's own "use cloud for this run" control (or a
 *   routing change), so the run falls back to a cloud job;
 * - `supersededSignal` - a newer run for the same page took over, so this run is
 *   cancelled outright: no error banner and no cloud job, or the two runs would
 *   race to write the same page.
 */
export type LocalRun = {
  cloudSwitchSignal: AbortSignal;
  supersededSignal: AbortSignal;
  /**
   * Release this run's hold on the page's local-inference state. Safe to call
   * from a superseded run: only the run that still owns the state clears it.
   */
  end: () => void;
};

export type LocalInferenceCallbacks = {
  /** Begin a local run, superseding whichever run the page was already doing. */
  startRun: (registryModelId: string, registryTag?: string) => LocalRun;
};

/**
 * A run that a newer run on the same page replaced. Carries no user-facing text:
 * the successor owns the outcome, so callers swallow it silently.
 */
export class RunSupersededError extends Error {
  constructor() {
    super("A newer run replaced this one.");
    this.name = "RunSupersededError";
  }
}

export function isRunSupersededError(error: unknown): boolean {
  return error instanceof RunSupersededError;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
