import { describe, expect, it } from "vitest";

import { ApiError } from "../api/errors";
import type { JobResponse } from "../api/client";
import {
  executionAnnouncement,
  isSubmissionRefusal,
  jobExecution,
  submissionRefusalExplanation,
} from "./executionTarget";
import { platformNoCapacityMessage } from "./platformMessages";

/**
 * A job exactly as `GET /jobs/{id}` returns it. Typed as `JobResponse` so the
 * generated contract - not this file - decides which fields exist.
 */
function job(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    id: "5f2b1c9e-0000-4000-8000-000000000001",
    type: "transcribe",
    status: "running",
    error: null,
    result: null,
    project_id: "5f2b1c9e-0000-4000-8000-000000000002",
    document_id: "5f2b1c9e-0000-4000-8000-000000000003",
    part_id: "5f2b1c9e-0000-4000-8000-000000000004",
    created_at: "2026-08-04T10:00:00Z",
    updated_at: "2026-08-04T10:00:00Z",
    started_at: null,
    completed_at: null,
    execution_target: "cloud",
    preferred_execution_target: "cloud",
    execution_target_substituted: false,
    execution: "cloud",
    ...overrides,
  };
}

describe("the announcement on a job", () => {
  it("names the inference host that will run it", () => {
    expect(
      executionAnnouncement(
        jobExecution(job({ execution_target: "local", status: "pending" })),
      ),
    ).toBe("Running on your computer.");

    expect(executionAnnouncement(jobExecution(job()))).toBe(
      "Running in the cloud.",
    );
  });

  it("says plainly when the preferred host was substituted", () => {
    const announcement = executionAnnouncement(
      jobExecution(
        job({
          execution_target: "cloud",
          preferred_execution_target: "local",
          execution_target_substituted: true,
        }),
      ),
    );

    expect(announcement).toBe(
      "Running in the cloud. You asked for your computer, which had no capacity when this job was submitted.",
    );
  });

  it("substitutes in the other direction too, and says which host it lost", () => {
    expect(
      executionAnnouncement(
        jobExecution(
          job({
            execution_target: "local",
            preferred_execution_target: "cloud",
            execution_target_substituted: true,
          }),
        ),
      ),
    ).toBe(
      "Running on your computer. You asked for the cloud, which had no capacity when this job was submitted.",
    );
  });

  it("shows which host a failed job failed on", () => {
    expect(
      executionAnnouncement(
        jobExecution(
          job({
            status: "failed",
            error: "weights not found",
            execution_target: "local",
          }),
        ),
      ),
    ).toBe("Failed on your computer.");
  });

  it("keeps naming the host after the job has substituted and failed", () => {
    expect(
      executionAnnouncement(
        jobExecution(
          job({
            status: "failed",
            execution_target: "cloud",
            preferred_execution_target: "local",
            execution_target_substituted: true,
          }),
        ),
      ),
    ).toBe(
      "Failed in the cloud. You asked for your computer, which had no capacity when this job was submitted.",
    );
  });

  it("reads in the past tense once the job is over", () => {
    expect(
      executionAnnouncement(
        jobExecution(job({ status: "done", execution_target: "local" })),
      ),
    ).toBe("Ran on your computer.");
    expect(
      executionAnnouncement(jobExecution(job({ status: "cancelled" }))),
    ).toBe("Cancelled in the cloud.");
  });
});

describe("a refused submission", () => {
  it("carries the platform's own explanation, not a rewritten one", () => {
    const refusal = new ApiError(platformNoCapacityMessage(), 409);

    expect(isSubmissionRefusal(refusal)).toBe(true);
    expect(submissionRefusalExplanation(refusal)).toBe(
      platformNoCapacityMessage(),
    );
    // The sentence the researcher can act on: which hosts were checked, and
    // what to do about it.
    expect(submissionRefusalExplanation(refusal)).toMatch(
      /no inference host is available/i,
    );
    expect(submissionRefusalExplanation(refusal)).toMatch(/agent/i);
  });

  it("is not confused with an ordinary failure", () => {
    expect(submissionRefusalExplanation(new ApiError("boom", 500))).toBeNull();
    expect(submissionRefusalExplanation(new Error("boom"))).toBeNull();
    expect(submissionRefusalExplanation(null)).toBeNull();
  });
});
