import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const CATALOG = [
  {
    id: "seg-a",
    name: "kraken",
    provider: "kraken",
    task: "segment",
    artifact_ref: "registry://blla-segment?tag=stable",
    default_params: {},
    created_at: "2026-09-21T00:00:00Z",
  },
  {
    id: "htr-1",
    name: "htr",
    provider: "calamari",
    task: "transcribe",
    artifact_ref: "registry://htr-1?tag=stable",
    default_params: {},
    created_at: "2026-09-21T00:00:00Z",
  },
];

type BindingRow = {
  id: string;
  task: string;
  model_id: string;
  project_id: string;
  document_id: null;
  document_part_id: null;
  overrides: Record<string, unknown>;
  created_at: string;
};

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * The whole loop with the real API client: no project binding, run a job
 * with a chosen model, the control shows "Project default" without a reload.
 */
describe("DocumentWorkflowMenu automatic project default", () => {
  let rows: BindingRow[];

  beforeEach(() => {
    rows = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        if (method === "GET" && url.includes("/inference/models")) {
          return jsonResponse(CATALOG);
        }
        if (method === "GET" && url.includes("/model-bindings")) {
          return jsonResponse(rows);
        }
        if (method === "POST" && url.includes("/model-bindings")) {
          const body = JSON.parse(String(init?.body)) as {
            task: string;
            model_id: string;
          };
          const created: BindingRow = {
            id: `binding-${body.task}`,
            task: body.task,
            model_id: body.model_id,
            project_id: "proj-r2",
            document_id: null,
            document_part_id: null,
            overrides: {},
            created_at: "2026-09-21T00:00:00Z",
          };
          rows = [...rows.filter((row) => row.task !== body.task), created];
          return jsonResponse(created, 201);
        }
        if (method === "POST" && url.includes("/jobs/transcribe")) {
          return jsonResponse({ queued: 1, skipped: 0, jobs: [] });
        }
        throw new Error(`unexpected fetch ${method} ${url}`);
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the quiet state after the first run, without a reload", async () => {
    render(
      <DocumentWorkflowMenu
        projectId="proj-r2"
        documentId="document-1"
        counts={{ total: 3, reviewed: 0, unsegmented: 1, unpaired: 2 }}
        onJobsQueued={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));

    const select = await screen.findByRole("combobox", {
      name: "HTR transcription model",
    });
    await waitFor(() => expect(select).toHaveValue("htr-1"));
    expect(screen.queryByText("Project default")).toBeNull();

    fireEvent.click(
      screen.getByRole("menuitem", { name: /transcribe unpaired pages/i }),
    );

    // A successful run closes the menu; reopening must show the quiet state
    // with no reload in between.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /workflow/i })).toHaveAttribute(
        "aria-expanded",
        "false",
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));

    await waitFor(() =>
      expect(screen.getAllByText("Project default")).toHaveLength(1),
    );
    expect(
      screen.getByRole("combobox", { name: "HTR transcription model" }),
    ).toHaveValue("htr-1");
  });
});
