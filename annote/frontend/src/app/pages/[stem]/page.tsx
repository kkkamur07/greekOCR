import PageEditor from "@/components/PageEditor";
import { fetchPages } from "@/lib/api";

interface PageProps {
  params: Promise<{ stem: string }>;
}

export default async function PageRoute({ params }: PageProps) {
  const { stem } = await params;
  let initialDirty = false;
  try {
    const data = await fetchPages();
    initialDirty = data.pages.find((p) => p.stem === stem)?.export_dirty ?? false;
  } catch {
    /* editor still loads; dirty defaults false */
  }
  return <PageEditor stem={stem} initialDirty={initialDirty} />;
}
