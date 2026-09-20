import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

const listInferenceModels = vi.fn();
const enqueueDocumentSegment = vi.fn();
const enqueueDocumentTranscribe = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    enqueueDocumentSegment: (...args: unknown[]) =>
      enqueueDocumentSegment(...args),
    enqueueDocumentTranscribe: (...args: unknown[]) =>
      enqueueDocumentTranscribe(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const COUNTS = { total: 3, reviewed: 0, unsegmented: 1, unpaired: 0 };
const SEGMENT_MODELS = [
  { id: "seg-a", task: "segment", name: "blla-segment" },
  { id: "seg-b", task: "segment", name: "pp-ocr" },
  { id: "htr-1", task: "transcribe", name: "htr" },
];

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

describe("DocumentWorkflowMenu segment picker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(SEGMENT_MODELS);
    enqueueDocumentSegment.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
  });

  it("sends model_id null for Default and the id for a chosen model", async () => {
    openMenu();

    await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    const options = screen.getAllByRole("option").map((option) => ({
      value: (option as HTMLOptionElement).value,
      text: option.textContent,
    }));
    expect(options).toEqual([
      { value: "", text: "Default" },
      { value: "seg-a", text: "blla-segment" },
      { value: "seg-b", text: "pp-ocr" },
    ]);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: null },
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    const reopenedSelect = screen.getByRole("combobox", {
      name: "Segmentation model",
    });
    fireEvent.change(reopenedSelect, { target: { value: "seg-b" } });
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
  });

  it("names the chosen model in the re-segment confirm, and the default when none is chosen", async () => {
    openMenu();
    await screen.findByRole("combobox", { name: "Segmentation model" });

    fireEvent.click(
      screen.getByRole("menuitem", { name: /re-segment every page/i }),
    );
    expect(screen.getByText(/runs with the default model/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("menuitem", { name: "Cancel" }));
    fireEvent.change(
      screen.getByRole("combobox", { name: "Segmentation model" }),
      { target: { value: "seg-b" } },
    );
    fireEvent.click(
      screen.getByRole("menuitem", { name: /re-segment every page/i }),
    );
    expect(screen.getByText(/runs with pp-ocr/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("menuitem", { name: /yes, re-segment/i }));
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "all", model_id: "seg-b" },
      ),
    );
  });

  it("leaves only Default and still segments when the catalog request fails", async () => {
    listInferenceModels.mockRejectedValue(new Error("offline"));
    openMenu();

    await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    expect(screen.getAllByRole("option")).toHaveLength(1);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: null },
      ),
    );
  });
});
