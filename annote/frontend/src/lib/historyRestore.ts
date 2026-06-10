import type { HistorySnapshotSummary } from "@/types/api";

export function latestHistorySnapshotId(snapshots: HistorySnapshotSummary[]): string | null {
  if (snapshots.length === 0) return null;
  let latest = snapshots[0]!;
  for (const snapshot of snapshots) {
    if (snapshot.timestamp > latest.timestamp) {
      latest = snapshot;
    }
  }
  return latest.id;
}
