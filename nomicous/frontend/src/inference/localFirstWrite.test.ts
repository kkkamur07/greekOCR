import { describe, expect, it, vi } from "vitest";
import { runLocalFirstWrite } from "./localFirstWrite";
import { isRunSupersededError } from "./localInferenceCallbacks";

function abortedRun() {
  const superseded = new AbortController();
  superseded.abort();
  return {
    cloudSwitchSignal: new AbortController().signal,
    supersededSignal: superseded.signal,
    end: () => {},
  };
}

function liveRun() {
  return {
    cloudSwitchSignal: new AbortController().signal,
    supersededSignal: new AbortController().signal,
    end: () => {},
  };
}

const passThroughTracker = <R,>(run: (signal: AbortSignal) => Promise<R>) =>
  run(new AbortController().signal);

describe("runLocalFirstWrite", () => {
  it("does not touch the cloud when the local write succeeds", async () => {
    const runInCloud = vi.fn();

    const outcome = await runLocalFirstWrite({
      cloudEnabled: true,
      trackLocalTask: passThroughTracker,
      runLocally: async () => "stored locally",
      runInCloud,
    });

    expect(outcome).toEqual({ source: "local", result: "stored locally" });
    expect(runInCloud).not.toHaveBeenCalled();
  });

  it("falls back to the cloud when the local write fails for a non-abort reason", async () => {
    const outcome = await runLocalFirstWrite({
      cloudEnabled: true,
      trackLocalTask: passThroughTracker,
      runLocally: async () => {
        throw new Error("WEIGHTS_UNAVAILABLE");
      },
      runInCloud: async () => "stored in cloud",
    });

    expect(outcome).toEqual({ source: "cloud", result: "stored in cloud" });
  });

  it("does not run in the cloud when the user cancels the local job", async () => {
    const runInCloud = vi.fn();
    const cancelled = new AbortController();
    cancelled.abort();

    await expect(
      runLocalFirstWrite({
        cloudEnabled: true,
        trackLocalTask: async (run) => {
          await run(cancelled.signal).catch(() => undefined);
          throw new DOMException("Local job cancelled", "AbortError");
        },
        runLocally: async () => "unreachable",
        runInCloud,
      }),
    ).rejects.toThrow("Local job cancelled");
    expect(runInCloud).not.toHaveBeenCalled();
  });

  it("cancels a superseded run outright instead of racing it in the cloud", async () => {
    const runInCloud = vi.fn();

    const failure = await runLocalFirstWrite({
      cloudEnabled: true,
      trackLocalTask: passThroughTracker,
      runLocally: async ({ reportRun }) => {
        reportRun(abortedRun());
        throw new DOMException("The operation was aborted.", "AbortError");
      },
      runInCloud,
    }).catch((error: unknown) => error);

    expect(isRunSupersededError(failure)).toBe(true);
    expect(runInCloud).not.toHaveBeenCalled();
  });

  it("reports an actionable error instead of using the cloud under local-only routing", async () => {
    const runInCloud = vi.fn();

    await expect(
      runLocalFirstWrite({
        cloudEnabled: false,
        trackLocalTask: passThroughTracker,
        runLocally: async ({ reportRun }) => {
          reportRun(liveRun());
          throw new Error("helper crashed");
        },
        runInCloud,
      }),
    ).rejects.toThrow(/Local only/);
    expect(runInCloud).not.toHaveBeenCalled();
  });

  it("attempts each path at most once, so a failing cloud write is not retried", async () => {
    const runLocally = vi.fn(async () => {
      throw new Error("helper crashed");
    });
    const runInCloud = vi.fn(async () => {
      throw new Error("cloud rejected the job");
    });

    await expect(
      runLocalFirstWrite({
        cloudEnabled: true,
        trackLocalTask: passThroughTracker,
        runLocally,
        runInCloud,
      }),
    ).rejects.toThrow("cloud rejected the job");

    expect(runLocally).toHaveBeenCalledTimes(1);
    expect(runInCloud).toHaveBeenCalledTimes(1);
  });
});
