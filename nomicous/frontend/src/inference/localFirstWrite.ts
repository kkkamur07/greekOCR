import { RunSupersededError, type LocalRun } from "./localInferenceCallbacks";

export type LocalFirstWriteResult<T> = {
  /** Which path produced `result`, for the sentence the page reports. */
  source: "local" | "cloud";
  result: T;
};

export type LocalFirstWriteOptions<T> = {
  /** The jobs-panel wrapper. Only an abort on the signal it owns is a user cancellation. */
  trackLocalTask: <R>(run: (signal: AbortSignal) => Promise<R>) => Promise<R>;
  /**
   * Runs the helper and persists what it produced, and nothing more. Everything
   * in here is part of "did the local run produce a saved result"; a failure
   * means the cloud still has to do the work.
   */
  runLocally: (context: {
    signal: AbortSignal;
    reportRun: (run: LocalRun) => void;
  }) => Promise<T>;
  /** Reached only from a failure of `runLocally`. */
  runInCloud: () => Promise<T>;
};

/**
 * A write attempted on the local helper first, falling back to the cloud.
 *
 * The fallback decision is taken here and nowhere else, and the only failure it
 * can ever observe is a failure of `runLocally`. There is deliberately **no
 * refresh parameter**: reloading the page to show what was written is the
 * caller's business, it happens strictly after this function has returned, and
 * so there is no control-flow path from a failing read back into `runInCloud`.
 *
 * That is the whole point. The bug this replaces was a `try` block wide enough
 * to contain both the write and the cosmetic reload that followed it, so a blip
 * on the reload - after a transcription or a segmentation was already stored -
 * re-ran the entire page in the cloud, paying twice and overwriting saved work.
 * Keeping the reload out of this function's signature is what makes that
 * unexpressible rather than merely commented against.
 *
 * Throws:
 * - the original `AbortError` when the *user* cancelled, so nothing continues
 *   silently in the cloud;
 * - `RunSupersededError` when a newer run for the same page took over, so this
 *   one is dropped outright rather than racing its successor.
 *
 * Every other local failure falls through to the cloud. There is no mode in
 * which it may not: `local_only` was retired by ADR 0002, and with it the only
 * way a write could end with neither host having done the work.
 */
export async function runLocalFirstWrite<T>({
  trackLocalTask,
  runLocally,
  runInCloud,
}: LocalFirstWriteOptions<T>): Promise<LocalFirstWriteResult<T>> {
  // The signal the jobs panel owns. Only an abort on *this* signal is a user
  // cancellation; the run's own signals say why else it stopped.
  let userCancelSignal: AbortSignal | undefined;
  // Held here rather than inside the run, so it stays readable after the run it
  // belongs to has already unwound.
  let supersededSignal: AbortSignal | undefined;

  try {
    const result = await trackLocalTask((signal) => {
      userCancelSignal = signal;
      return runLocally({
        signal,
        reportRun: (run) => {
          supersededSignal = run.supersededSignal;
        },
      });
    });
    return { source: "local", result };
  } catch (error) {
    // A cancellation the user asked for must stop here, never continue silently
    // in the cloud.
    if (userCancelSignal?.aborted) throw error;
    // A newer run for this page owns the outcome now. Drop this one outright -
    // no banner, and above all no cloud job to race with it.
    if (supersededSignal?.aborted) throw new RunSupersededError();
    // Any other local failure (weights missing, helper crash, 503, …) falls
    // through to the cloud.
  }

  return { source: "cloud", result: await runInCloud() };
}
