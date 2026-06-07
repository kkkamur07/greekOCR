"use client";

import type { PairingProgress } from "@/types/api";
import { formatPairingProgress, isPairingComplete } from "@/lib/pairingProgress";

interface PairingProgressBarProps {
  progress: PairingProgress;
}

export default function PairingProgressBar({ progress }: PairingProgressBarProps) {
  const complete = isPairingComplete(progress);

  return (
    <div
      className={`flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs ${
        complete ? "text-emerald-700" : "text-gray-600"
      }`}
      title={formatPairingProgress(progress)}
    >
      <span>
        <span className="font-medium text-violet-700">{progress.paired_count}</span> paired
      </span>
      <span className="text-gray-300">·</span>
      <span>
        <span className={`font-medium ${progress.unpaired_count > 0 ? "text-amber-700" : "text-gray-500"}`}>
          {progress.unpaired_count}
        </span>{" "}
        unpaired
      </span>
      {progress.text_line_count > 0 && (
        <>
          <span className="text-gray-300">·</span>
          <span>
            <span
              className={`font-medium ${progress.unused_line_count > 0 ? "text-sky-700" : "text-gray-500"}`}
            >
              {progress.unused_line_count}
            </span>{" "}
            unused line{progress.unused_line_count === 1 ? "" : "s"}
          </span>
        </>
      )}
      {complete && progress.paired_count > 0 && (
        <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-800">
          complete
        </span>
      )}
    </div>
  );
}
