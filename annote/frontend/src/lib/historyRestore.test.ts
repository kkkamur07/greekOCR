import { describe, expect, it } from "vitest";

import type { HistorySnapshotSummary } from "@/types/api";

import { latestHistorySnapshotId } from "./historyRestore";

const SNAPSHOTS: HistorySnapshotSummary[] = [
  {
    id: "older",
    timestamp: "2026-06-07T12:00:00+00:00",
    reason: "timed",
    pairing_progress_percent: 25,
  },
  {
    id: "newer",
    timestamp: "2026-06-07T12:05:00+00:00",
    reason: "milestone_50",
    pairing_progress_percent: 50,
  },
];

describe("latestHistorySnapshotId", () => {
  it("returns the most recent snapshot id by timestamp", () => {
    expect(latestHistorySnapshotId(SNAPSHOTS)).toBe("newer");
  });

  it("returns null when there are no snapshots", () => {
    expect(latestHistorySnapshotId([])).toBeNull();
  });
});
