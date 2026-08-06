import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { MessageInstance } from "antd/es/message/interface";

import { ApiError } from "../../api/errors";
import { registerToastApi } from "../../components/ui/toast";
import { platformNoCapacityMessage } from "../../inference/platformMessages";
import {
  DOCUMENT,
  flushPageEditorEffects,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
  seedExecutionPreference,
} from "./testSupport";

/**
 * Records what `toast` would have put on screen.
 *
 * The themed `message` instance is normally handed to `toast` at mount by
 * `ToastBridge`, which these tests do not render - so without a stand-in every
 * toast is silently dropped and "this did not become a toast" would pass for
 * the wrong reason. The returned array is the assertion's evidence.
 */
function recordedToasts(): string[] {
  const shown: string[] = [];
  const record = (content: unknown) => {
    shown.push(String(content));
    return (() => {}) as unknown as ReturnType<MessageInstance["success"]>;
  };
  registerToastApi({
    success: record,
    error: record,
    info: record,
    warning: record,
    loading: record,
    open: record,
    destroy: () => {},
  } as unknown as MessageInstance);
  return shown;
}

afterEach(() => registerToastApi(null));

const LINE = {
  id: "line-1",
  part_id: "part-1",
  block_id: null,
  order: 0,
  kind: "polygon" as const,
  points: [
    [10, 10],
    [50, 10],
    [50, 30],
    [10, 30],
  ],
  source: "manual" as const,
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [],
  created_at: "2026-06-16T10:00:00Z",
};

const TRANSCRIBE_MODEL = {
  id: "model-1",
  name: "kraken-transcribe-default",
  provider: "kraken",
  task: "transcribe",
  artifact_ref: "registry://greek-calamari-v1",
  default_params: {},
  created_at: "2026-06-16T10:00:00Z",
};

/** A job as `GET /jobs/{id}` returns it, including its **execution target**. */
function jobResponse(overrides: Record<string, unknown> = {}) {
  return {
    id: "job-ocr-1",
    type: "transcribe",
    status: "done",
    payload: {},
    result: { transcription_id: "model-2", lines: [] },
    error: null,
    document_id: "doc-1",
    document_part_id: "part-1",
    created_at: "2026-06-16T10:00:00Z",
    updated_at: "2026-06-16T10:00:00Z",
    started_at: "2026-06-16T10:00:00Z",
    completed_at: "2026-06-16T10:00:00Z",
    execution_target: "cloud",
    preferred_execution_target: "cloud",
    execution_target_substituted: false,
    execution: "cloud",
    ...overrides,
  };
}

function loadedPageWithOneSegment() {
  mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  mockedApi.listInferenceModels.mockResolvedValue([TRANSCRIBE_MODEL]);
  mockedApi.listPartLines.mockResolvedValue([LINE]);
}

async function runOcrOnTheSelectedSegment() {
  fireEvent.click(await screen.findByLabelText(/^Segment 1/));
  fireEvent.click(
    screen.getByRole("button", { name: /re-run ocr on segment 1/i }),
  );
}

async function openBackgroundJobs() {
  fireEvent.click(
    await screen.findByRole("button", { name: /background job/i }),
  );
}

describe("the account-level host preference", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("persists and round-trips through the account, not the browser", async () => {
    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /editor settings/i }),
    );
    const setting = screen.getByRole("checkbox", {
      name: /use my computer when it is available/i,
    });
    expect(setting).not.toBeChecked();

    fireEvent.click(setting);

    await waitFor(() => {
      expect(mockedApi.setExecutionPreference).toHaveBeenCalledWith(true);
    });
    await waitFor(() => expect(setting).toBeChecked());

    // A fresh mount reads the account again. Were the setting held in this
    // browser rather than on the account, this is where it would be lost.
    await flushPageEditorEffects();
    cleanup();
    renderPageEditor();
    fireEvent.click(
      await screen.findByRole("button", { name: /editor settings/i }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("checkbox", {
          name: /use my computer when it is available/i,
        }),
      ).toBeChecked(),
    );
  });

  it("offers no per-job execution target control alongside the run actions", async () => {
    seedExecutionPreference(true);
    loadedPageWithOneSegment();
    renderPageEditor();

    await screen.findByLabelText(/^Segment 1/);
    // Every control that could pick a host for one run, by any of the names the
    // retired three-mode picker used.
    expect(screen.queryByRole("radio", { name: /cloud/i })).toBeNull();
    expect(screen.queryByRole("radio", { name: /local/i })).toBeNull();
    expect(screen.queryByLabelText(/run this in the cloud/i)).toBeNull();
    expect(screen.queryByText(/local only/i)).toBeNull();
  });
});

describe("the announcement on a job", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    loadedPageWithOneSegment();
    mockedApi.enqueueTranscribePart.mockResolvedValue({ job_id: "job-ocr-1" });
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("states the inference host the job was given", async () => {
    mockedApi.getJob.mockResolvedValue(jobResponse());

    renderPageEditor();
    await runOcrOnTheSelectedSegment();
    await openBackgroundJobs();

    expect(await screen.findByText("Ran in the cloud.")).toBeTruthy();
  });

  it("states a substituted host on the job rather than in a toast", async () => {
    const toasts = recordedToasts();
    mockedApi.getJob.mockResolvedValue(
      jobResponse({
        execution_target: "cloud",
        preferred_execution_target: "local",
        execution_target_substituted: true,
      }),
    );

    renderPageEditor();
    await runOcrOnTheSelectedSegment();
    await openBackgroundJobs();

    const announcement = await screen.findByText(
      /you asked for your computer, which had no capacity/i,
    );
    // It stays on the job. A toast would be gone before a researcher who
    // looked away could read where their work went.
    expect(announcement).toBeTruthy();
    expect(toasts.filter((shown) => /had no capacity/i.test(shown))).toEqual(
      [],
    );
  });

  it("shows which host a failed job failed on", async () => {
    mockedApi.getJob.mockResolvedValue(
      jobResponse({
        status: "failed",
        error: "weights could not be loaded",
        execution_target: "local",
        preferred_execution_target: "local",
      }),
    );

    renderPageEditor();
    await runOcrOnTheSelectedSegment();
    await openBackgroundJobs();

    expect(await screen.findByText("Failed on your computer.")).toBeTruthy();
  });
});

describe("a submission the platform refuses", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    loadedPageWithOneSegment();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("explains that no inference host had capacity, and keeps the explanation on screen", async () => {
    const toasts = recordedToasts();
    mockedApi.enqueueTranscribePart.mockRejectedValue(
      new ApiError(platformNoCapacityMessage(), 409),
    );

    renderPageEditor();
    await runOcrOnTheSelectedSegment();

    const explanation = await screen.findByText(platformNoCapacityMessage());
    expect(explanation).toBeTruthy();
    // Not the generic error toast the ordinary failure path uses.
    expect(toasts).not.toContain(platformNoCapacityMessage());
  });
});
