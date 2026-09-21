import { waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, subscribeProjectDefaultWritten } from "./client";
import { ApiError } from "./errors";

type FetchCall = {
  url: string;
  method: string;
};

const BATCH_JOB = { queued: 1, skipped: 0, jobs: [] };
const PART_JOB = {
  job_id: "job-1",
  execution_target: "cloud",
  execution_target_substituted: false,
  preferred_execution_target: "cloud",
};
const BINDING = {
  id: "binding-1",
  task: "transcribe",
  model_id: "htr-syriac",
  project_id: "proj-a",
  document_id: null,
  document_part_id: null,
  overrides: {},
  created_at: "2026-09-21T00:00:00Z",
};

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("enqueue wrappers and the automatic project default", () => {
  const calls: FetchCall[] = [];
  let jobBehavior: "ok" | "reject" = "ok";
  let bindingsBehavior: "empty" | "hang" = "empty";

  beforeEach(() => {
    calls.length = 0;
    jobBehavior = "ok";
    bindingsBehavior = "empty";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        calls.push({ url, method });
        if (method === "POST" && /\/jobs\/(segment|transcribe)$/.test(url)) {
          if (jobBehavior === "reject") {
            return jsonResponse({ error: { message: "busy" } }, 500);
          }
          return jsonResponse(BATCH_JOB);
        }
        if (
          method === "POST" &&
          /\/(segment|transcribe)$/.test(url) &&
          !url.includes("/jobs/")
        ) {
          return jsonResponse(PART_JOB);
        }
        if (method === "GET" && url.includes("/model-bindings")) {
          if (bindingsBehavior === "hang") {
            return new Promise<Response>(() => {});
          }
          return jsonResponse([]);
        }
        if (method === "POST" && url.includes("/model-bindings")) {
          return jsonResponse(BINDING);
        }
        throw new Error(`unexpected fetch ${method} ${url}`);
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends the job request before any bindings request", async () => {
    const response = await api.enqueueDocumentTranscribe("proj-a", "doc-1", {
      scope: "unpaired",
      model_id: "htr-syriac",
    });
    expect(response).toEqual(BATCH_JOB);

    await waitFor(() =>
      expect(
        calls.filter(
          (call) =>
            call.method === "POST" && call.url.includes("model-bindings"),
        ),
      ).toHaveLength(1),
    );
    expect(calls.map((call) => `${call.method} ${call.url}`)).toEqual([
      expect.stringMatching(/^POST .*\/jobs\/transcribe$/),
      expect.stringMatching(/^GET .*\/model-bindings$/),
      expect.stringMatching(/^POST .*\/model-bindings$/),
    ]);
  });

  it("returns the job response even when the bindings list never resolves", async () => {
    bindingsBehavior = "hang";
    const response = await api.enqueueDocumentTranscribe("proj-b", "doc-1", {
      scope: "unpaired",
      model_id: "htr-syriac",
    });
    expect(response).toEqual(BATCH_JOB);
    // The background write is still stuck on the list: no binding POST went out.
    await Promise.resolve();
    expect(calls).toHaveLength(2);
  });

  it("writes no default when the job request is rejected", async () => {
    jobBehavior = "reject";
    await expect(
      api.enqueueDocumentTranscribe("proj-c", "doc-1", {
        scope: "unpaired",
        model_id: "htr-syriac",
      }),
    ).rejects.toBeInstanceOf(ApiError);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ method: "POST" });
  });

  it("orders segment batch jobs the same way", async () => {
    const response = await api.enqueueDocumentSegment("proj-d", "doc-1", {
      scope: "unsegmented",
      model_id: "seg-a",
    });
    expect(response).toEqual(BATCH_JOB);

    await waitFor(() =>
      expect(
        calls.filter(
          (call) =>
            call.method === "POST" && call.url.includes("model-bindings"),
        ),
      ).toHaveLength(1),
    );
    expect(calls[0]).toMatchObject({ method: "POST" });
    expect(calls[0].url).toMatch(/\/jobs\/segment$/);
  });

  it("announces each automatic write to subscribers", async () => {
    const seen: { projectId: string; task: string }[] = [];
    const unsubscribe = subscribeProjectDefaultWritten((info) => {
      seen.push({ projectId: info.projectId, task: info.task });
    });
    try {
      await api.enqueueDocumentSegment("proj-f", "doc-1", {
        scope: "unsegmented",
        model_id: "seg-a",
      });
      await waitFor(() =>
        expect(seen).toEqual([{ projectId: "proj-f", task: "segment" }]),
      );
    } finally {
      unsubscribe();
    }
  });

  it("orders part jobs the same way", async () => {
    const response = await api.enqueueTranscribePart(
      "proj-e",
      "doc-1",
      "part-1",
      {
        model_id: "htr-syriac",
      },
    );
    expect(response).toEqual(PART_JOB);

    await waitFor(() =>
      expect(
        calls.filter(
          (call) =>
            call.method === "POST" && call.url.includes("model-bindings"),
        ),
      ).toHaveLength(1),
    );
    expect(calls[0].url).toMatch(/\/parts\/part-1\/transcribe$/);
  });
});
