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

function errorDetail(error: unknown): string {
  const message = error instanceof Error ? error.message.trim() : "";
  return message ? ` (${message})` : "";
}

/**
 * Shown when a local run fails and "Local only" forbids the cloud retry that
 * "Automatic" would have made instead.
 */
export function localOnlyRunFailedMessage(error: unknown): string {
  return `Running on this computer failed${errorDetail(error)}. Inference is set to "Local only", so it was not sent to the cloud - switch to "Automatic" to allow the cloud, or restart the Nomicous Inference Helper and try again.`;
}

/** Shown when "Local only" is set but no helper can serve the model. */
export function localOnlyUnavailableMessage(): string {
  return 'Inference is set to "Local only" and the Nomicous Inference Helper is not available for this model on this computer. Start the helper, or switch to "Automatic" to run this in the cloud.';
}
