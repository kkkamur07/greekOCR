import { afterEach, describe, expect, it, vi } from "vitest";

import type { JobResponse } from "../api/client";
import { subscribeToJob } from "./jobSubscription";

function job(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    id: "job-1",
    type: "pipeline",
    status: "done",
    payload: {},
    result: { ok: true },
    error: null,
    user_id: "user-1",
    document_id: null,
    document_part_id: null,
    created_at: "2026-07-09T10:00:00Z",
    updated_at: "2026-07-09T10:00:01Z",
    started_at: "2026-07-09T10:00:00Z",
    completed_at: "2026-07-09T10:00:01Z",
    ...overrides,
  };
}

describe("subscribeToJob", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("shares one stalled SSE stream and falls back to polling", async () => {
    vi.useFakeTimers();
    const stream = new ReadableStream<Uint8Array>({
      start() {
        // Keep the stream open without sending an event or heartbeat.
      },
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(stream, { status: 200 }));
    const getJob = vi.fn().mockResolvedValue(job());
    const firstListener = vi.fn();
    const secondListener = vi.fn();

    const first = subscribeToJob("job-1", {
      eventsUrl: "http://localhost:8000/jobs/job-1/events",
      getJob,
      intervalMs: 100,
      noEventTimeoutMs: 50,
      onUpdate: firstListener,
    });
    const second = subscribeToJob("job-1", {
      eventsUrl: "http://localhost:8000/jobs/job-1/events",
      getJob,
      intervalMs: 100,
      noEventTimeoutMs: 50,
      onUpdate: secondListener,
    });

    await vi.advanceTimersByTimeAsync(50);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(getJob).toHaveBeenCalledWith("job-1");
    expect(firstListener).toHaveBeenCalledWith(job());
    expect(secondListener).toHaveBeenCalledWith(job());
    first();
    second();
  });
});
