import Link from "next/link";

import PairingProgressBar from "@/components/PairingProgressBar";
import { displayPageName, formatPageTitle } from "@/lib/pageName";
import type { PageSummary } from "@/types/api";

function pageInitial(stem: string): string {
  const match = displayPageName(stem).match(/[a-zA-Zα-ωΑ-Ω0-9]/);
  return match ? match[0].toUpperCase() : "?";
}

export default function PageCard({ page }: { page: PageSummary }) {
  const title = formatPageTitle(page.stem);
  const label = displayPageName(page.stem);

  return (
    <li>
      <Link
        href={`/pages/${encodeURIComponent(page.stem)}`}
        className="group flex items-center gap-3 rounded-xl border border-gray-200 bg-white px-4 py-3 shadow-sm transition-all hover:border-gray-300 hover:shadow-md"
      >
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-sm font-semibold text-gray-600 group-hover:bg-gray-900 group-hover:text-white">
          {pageInitial(page.stem)}
        </span>

        <span className="min-w-0 flex-1">
          <span className="truncate text-sm font-medium text-gray-900" title={title}>
            {label}
          </span>
          <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-gray-500">
            <span>
              {page.segment_count} segment{page.segment_count === 1 ? "" : "s"}
            </span>
            {page.has_transcription && <span>· transcription</span>}
            {page.export_dirty && (
              <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-amber-800">needs export</span>
            )}
            {page.locked && (
              <span className="rounded-full bg-slate-200 px-1.5 py-0.5 text-slate-700">locked</span>
            )}
          </span>
          {page.segment_count > 0 && (
            <span className="mt-1 block">
              <PairingProgressBar progress={page.pairing} />
            </span>
          )}
        </span>

        <span className="shrink-0 text-gray-300 transition-transform group-hover:translate-x-0.5 group-hover:text-gray-500">
          →
        </span>
      </Link>
    </li>
  );
}
