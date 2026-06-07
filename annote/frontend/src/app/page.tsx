import PageCard from "@/components/PageCard";
import PageImport from "@/components/PageImport";
import { fetchPages } from "@/lib/api";
import type { PageSummary } from "@/types/api";

export const dynamic = "force-dynamic";

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
            Cannot reach the API — check that the backend is running on port 5050
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
