import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import HistoryPanel from "./HistoryPanel";
import type { HistorySnapshotSummary } from "@/types/api";

afterEach(cleanup);

const SNAPSHOTS: HistorySnapshotSummary[] = [
  {
    id: "snap-1",
    timestamp: "2026-06-07T12:00:00+00:00",
    reason: "milestone_50",
    pairing_progress_percent: 50,
  },
  {
    id: "snap-2",
    timestamp: "2026-06-07T12:05:00+00:00",
    reason: "lock",
    pairing_progress_percent: 100,
  },
];

describe("HistoryPanel", () => {
  it("lists snapshots with reason labels", () => {
    render(<HistoryPanel snapshots={SNAPSHOTS} restoring={false} onRestore={vi.fn()} />);

    expect(screen.getByText(/50% milestone/i)).toBeInTheDocument();
    expect(screen.getByText(/locked/i)).toBeInTheDocument();
  });

  it("disables restore while the page is locked", () => {
    render(<HistoryPanel snapshots={SNAPSHOTS} restoring={false} locked onRestore={vi.fn()} />);

    expect(screen.getByRole("button", { name: /restore snap-1/i })).toBeDisabled();
  });

  it("confirms before restoring a snapshot", () => {
    const onRestore = vi.fn();
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<HistoryPanel snapshots={SNAPSHOTS} restoring={false} onRestore={onRestore} />);

    fireEvent.click(screen.getByRole("button", { name: /restore snap-2/i }));

    expect(onRestore).toHaveBeenCalledWith("snap-2");
  });
});
