import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  BackgroundJobsProvider,
  useBackgroundJobs,
} from "./BackgroundJobsContext";

vi.mock("../hooks/useJobPolling", () => ({
  useJobPolling: () => undefined,
}));

vi.mock("../components/ui/toast", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

function LocalCancelProbe() {
  const { jobs, trackLocalTask, cancelJob } = useBackgroundJobs();
  return (
    <div>
      <output data-testid="status">{jobs[0]?.status ?? "none"}</output>
      <button
        type="button"
        onClick={() => {
          void trackLocalTask(
            { label: "Local segment", kind: "segmentation" },
            async (signal) => {
              await new Promise<void>((resolve, reject) => {
                const onAbort = () =>
                  reject(new DOMException("Aborted", "AbortError"));
                if (signal.aborted) {
                  onAbort();
                  return;
                }
                signal.addEventListener("abort", onAbort, { once: true });
                window.setTimeout(() => resolve(), 500);
              });
              return "persisted";
            },
          ).catch(() => undefined);
        }}
      >
        Start
      </button>
      <button
        type="button"
        onClick={() => {
          const jobId = jobs[0]?.id;
          if (jobId) void cancelJob(jobId);
        }}
      >
        Cancel local
      </button>
    </div>
  );
}

describe("BackgroundJobsContext local cancel", () => {
  it("aborts the local task and keeps status cancelled instead of done", async () => {
    render(
      <BackgroundJobsProvider>
        <LocalCancelProbe />
      </BackgroundJobsProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => {
      expect(screen.getByTestId("status").textContent).toBe("running");
    });

    fireEvent.click(screen.getByRole("button", { name: "Cancel local" }));
    await waitFor(() => {
      expect(screen.getByTestId("status").textContent).toBe("cancelled");
    });
  });
});
