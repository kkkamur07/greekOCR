import { describe, expect, it } from "vitest";

import type { JobResponse } from "../../api/client";
import { jobStatusLabel } from "./jobProgress";

function job(partial: Partial<JobResponse>): JobResponse {
  return {
    id: "job-1",
    type: "transcribe",
    status: "pending",
    payload: {},
    result: null,
    error: null,
    document_id: null,
    document_part_id: null,
    user_id: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...partial,
  };
}

describe("jobProgress", () => {
  it("labels job statuses for the page editor queue", () => {
    expect(jobStatusLabel(job({ status: "pending" }))).toBe("Queued");
    expect(jobStatusLabel(job({ status: "running" }))).toBe("Starting…");
    expect(jobStatusLabel(job({ status: "waiting" }))).toBe("Processing…");
    expect(jobStatusLabel(job({ status: "done" }))).toBe("Complete");
    expect(jobStatusLabel(job({ status: "failed" }))).toBe("Failed");
  });
});
