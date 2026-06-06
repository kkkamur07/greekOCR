import Link from "next/link";
import PageImport from "@/components/PageImport";
import { fetchPages } from "@/lib/api";
import { displayPageName, formatPageTitle } from "@/lib/pageName";
import type { PageSummary } from "@/types/api";

export const dynamic = "force-dynamic";

function pageInitial(stem: string): string {
  const match = displayPageName(stem).match(/[a-zA-Zα-ωΑ-Ω0-9]/);
  return match ? match[0].toUpperCase() : "?";
}

function PageCard({ page }: { page: PageSummary }) {
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
          </span>
        </span>

        <span className="shrink-0 text-gray-300 transition-transform group-hover:translate-x-0.5 group-hover:text-gray-500">
          →
        </span>
      </Link>
    </li>
  );
}

export default async function HomePage() {
  let pages: PageSummary[] = [];
  let apiOffline = false;
  try {
    const data = await fetchPages();
    pages = data.pages;
  } catch {
    apiOffline = true;
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <div className="mx-auto max-w-md px-5 py-14 sm:py-20">
        <header className="text-center">
          <h1 className="text-2xl font-semibold tracking-tight text-gray-900">annote</h1>
          <p className="mt-1.5 text-sm text-gray-500">Segment lines and pair transcriptions</p>
        </header>

        {apiOffline && (
          <p className="mt-6 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-center text-xs text-amber-900">
            Backend not running — start <code className="font-mono">annote</code> on port 5050
          </p>
        )}

        <div className="mt-8">
          <PageImport />
        </div>

        {pages.length > 0 && (
          <section className="mt-10">
            <h2 className="mb-3 px-1 text-xs font-medium uppercase tracking-wide text-gray-400">
              Pages
            </h2>
            <ul className="space-y-2">
              {pages.map((page) => (
                <PageCard key={page.stem} page={page} />
              ))}
            </ul>
          </section>
        )}
      </div>
    </main>
  );
}
