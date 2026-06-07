"use client";

import type { HistorySnapshotSummary } from "@/types/api";

function reasonLabel(reason: string): string {
  if (reason === "timed") return "Timed snapshot";
  if (reason.startsWith("milestone_")) {
    const pct = reason.replace("milestone_", "");
    return `${pct}% milestone`;
  }
  if (reason === "lock") return "Locked";
  if (reason === "unlock") return "Unlocked";
  return reason;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

interface HistoryPanelProps {
  snapshots: HistorySnapshotSummary[];
  restoring: boolean;
  locked?: boolean;
  onRestore: (snapshotId: string) => void;
}

export default function HistoryPanel({ snapshots, restoring, locked = false, onRestore }: HistoryPanelProps) {
  if (snapshots.length === 0) {
    return <p className="text-sm text-gray-500">No history snapshots yet.</p>;
  }

  const ordered = [...snapshots].reverse();

  return (
    <ul className="max-h-48 space-y-2 overflow-y-auto text-sm">
      {ordered.map((snapshot) => (
        <li
          key={snapshot.id}
          className="flex items-start justify-between gap-2 rounded border border-gray-200 bg-white px-2.5 py-2"
        >
          <div className="min-w-0">
            <p className="font-medium text-gray-800">{reasonLabel(snapshot.reason)}</p>
            <p className="text-xs text-gray-500">{formatTimestamp(snapshot.timestamp)}</p>
            <p className="text-xs text-gray-400">{snapshot.pairing_progress_percent}% paired</p>
          </div>
          <button
            type="button"
            disabled={restoring || locked}
            aria-label={`Restore ${snapshot.id}`}
            title={locked ? "Unlock the page before restoring history" : undefined}
            onClick={() => {
              if (locked) return;
              if (!window.confirm(`Restore snapshot from ${reasonLabel(snapshot.reason)}?`)) return;
              onRestore(snapshot.id);
            }}
            className="shrink-0 rounded px-2 py-1 text-xs text-indigo-800 hover:bg-indigo-50 disabled:opacity-40"
          >
            Restore
          </button>
        </li>
      ))}
    </ul>
  );
}
