import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

const listInferenceModels = vi.fn();
const listProjectModelBindings = vi.fn();
const createProjectModelBinding = vi.fn();
const updateProjectModelBinding = vi.fn();
const enqueueDocumentSegment = vi.fn();
const enqueueDocumentTranscribe = vi.fn();
const error = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    listProjectModelBindings: (...args: unknown[]) =>
      listProjectModelBindings(...args),
    createProjectModelBinding: (...args: unknown[]) =>
      createProjectModelBinding(...args),
    updateProjectModelBinding: (...args: unknown[]) =>
      updateProjectModelBinding(...args),
    enqueueDocumentSegment: (...args: unknown[]) =>
      enqueueDocumentSegment(...args),
    enqueueDocumentTranscribe: (...args: unknown[]) =>
      enqueueDocumentTranscribe(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: (...args: unknown[]) => error(...args) },
}));

const COUNTS = { total: 3, reviewed: 0, unsegmented: 1, unpaired: 2 };
const CATALOG = [
  { id: "seg-a", task: "segment", name: "kraken" },
  { id: "seg-b", task: "segment", name: "pp-ocr" },
  { id: "htr-1", task: "transcribe", name: "greek" },
  { id: "htr-2", task: "transcribe", name: "syriac" },
];

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function openMenu() {
  render(
    <DocumentWorkflowMenu
      projectId="project-1"
      documentId="document-1"
      counts={COUNTS}
      onJobsQueued={() => {}}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
}

function segmentSelect(): HTMLSelectElement {
  return screen.getByRole("combobox", {
    name: "Segmentation model",
  }) as HTMLSelectElement;
}

function transcribeSelect(): HTMLSelectElement {
  return screen.getByRole("combobox", {
    name: "HTR transcription model",
  }) as HTMLSelectElement;
}

describe("DocumentWorkflowMenu sets the project default", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(CATALOG);
    listProjectModelBindings.mockResolvedValue([]);
    enqueueDocumentSegment.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
    enqueueDocumentTranscribe.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
  });

  it("saves the pick as the project default and says so", async () => {
    const write = deferred<unknown>();
    createProjectModelBinding.mockReturnValue(write.promise);
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    await waitFor(() => expect(screen.getByText("Saving…")).toBeTruthy());
    // Only the task being written is frozen; the other picker stays usable.
    expect(segmentSelect()).toBeDisabled();
    expect(transcribeSelect()).not.toBeDisabled();

    write.resolve({ id: "binding-1", task: "segment", model_id: "seg-b" });

    await waitFor(() =>
      expect(screen.getByText("Project default")).toBeTruthy(),
    );
    expect(createProjectModelBinding).toHaveBeenCalledWith("project-1", {
      task: "segment",
      model_id: "seg-b",
    });
    expect(segmentSelect()).toHaveValue("seg-b");
  });

  it("marks a stored default without writing anything", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "transcribe", model_id: "htr-2" },
    ]);
    openMenu();

    await waitFor(() => expect(transcribeSelect()).toHaveValue("htr-2"));
    expect(screen.getAllByText("Project default")).toHaveLength(1);
    expect(createProjectModelBinding).not.toHaveBeenCalled();
    expect(updateProjectModelBinding).not.toHaveBeenCalled();
  });

  it("returns the select to the stored default when the write fails", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    updateProjectModelBinding.mockRejectedValue(new ApiError("No access", 403));
    openMenu();
    await waitFor(() => expect(segmentSelect()).toHaveValue("seg-a"));

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    await waitFor(() => expect(error).toHaveBeenCalledWith("No access"));
    expect(segmentSelect()).toHaveValue("seg-a");
  });

  it("keeps the pick for the run when the project has no stored default", async () => {
    createProjectModelBinding.mockRejectedValue(new ApiError("No access", 403));
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    await waitFor(() => expect(error).toHaveBeenCalledWith("No access"));
    // Nothing stored to fall back to, so the run keeps the model it replaced.
    expect(segmentSelect()).toHaveValue("seg-a");
    expect(screen.queryByText("Project default")).toBeNull();
  });

  it("writes no default and claims none while the bindings cannot be read", async () => {
    listProjectModelBindings.mockRejectedValue(new ApiError("offline", 503));
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    // The pick still runs the job, it just does not speak for the project.
    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: "seg-b" },
      ),
    );
    expect(createProjectModelBinding).not.toHaveBeenCalled();
    expect(updateProjectModelBinding).not.toHaveBeenCalled();
    expect(screen.queryByText("Project default")).toBeNull();
    expect(error).not.toHaveBeenCalled();
  });
});
