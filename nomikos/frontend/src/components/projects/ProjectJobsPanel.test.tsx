import { render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ProjectJobsPanel } from "./ProjectJobsPanel";

const listProjectJobsPage = vi.fn();
const clearProjectJobHistory = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listProjectJobsPage: (...args: unknown[]) => listProjectJobsPage(...args),
    clearProjectJobHistory: (...args: unknown[]) =>
      clearProjectJobHistory(...args),
  },
}));

vi.mock("../../hooks/useJobPolling", () => ({
  useJobPolling: () => undefined,
}));

const doneJob = {
  id: "11111111-2222-3333-4444-555555555555",
  type: "segment",
  status: "done",
  execution_target: "cloud",
  preferred_execution_target: "cloud",
  execution_target_substituted: false,
  execution: "cloud",
  error: null,
  document_id: null,
  created_at: "2026-08-01T10:00:00Z",
};

describe("ProjectJobsPanel clear history", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listProjectJobsPage.mockResolvedValue({
      items: [doneJob],
      next_cursor: null,
    });
    clearProjectJobHistory.mockResolvedValue({ deleted: 1 });
  });

  it("clears only finished jobs after an explicit confirmation", async () => {
    render(<ProjectJobsPanel projectId="project-1" documents={[]} />);

    const trigger = await screen.findByRole("button", {
      name: /clear history/i,
    });
    fireEvent.click(trigger);

    expect(
      await screen.findByText(/only finished jobs.*are removed/i),
    ).toBeTruthy();
    expect(clearProjectJobHistory).not.toHaveBeenCalled();

    const confirmButtons = await screen.findAllByRole("button", {
      name: /clear history/i,
    });
    fireEvent.click(confirmButtons[confirmButtons.length - 1]);

    await waitFor(() => {
      expect(clearProjectJobHistory).toHaveBeenCalledWith("project-1");
    });
    await waitFor(() => {
      expect(listProjectJobsPage).toHaveBeenCalledTimes(2);
    });
  });

  it("offers nothing to clear when every job is still running", async () => {
    listProjectJobsPage.mockResolvedValue({
      items: [{ ...doneJob, status: "running" }],
      next_cursor: null,
    });

    render(<ProjectJobsPanel projectId="project-1" documents={[]} />);

    await screen.findByRole("button", { name: /jobs/i });
    expect(screen.queryByRole("button", { name: /clear history/i })).toBeNull();
  });
});
