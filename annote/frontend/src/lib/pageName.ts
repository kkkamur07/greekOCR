function stemParts(stem: string): string[] {
  const withoutExt = stem.replace(/\.(jpe?g|png|webp|tiff?)$/i, "");
  return withoutExt.trim().replace(/^_+|_+$/g, "").split(/_+/).filter(Boolean);
}

/** Full cleaned title for tooltips. */
export function formatPageTitle(stem: string): string {
  const parts = stemParts(stem);
  if (parts.length === 0) return "page";
  return parts.join(" ");
}

/** Compact slug for lists and the toolbar. */
export function pageSlug(stem: string): string {
  const parts = stemParts(stem);
  if (parts.length === 0) return "page";
  if (parts.length === 1) return parts[0].toLowerCase();

  const last = parts[parts.length - 1];
  const pageSuffix = /^\d+[a-z]?$/i.test(last) || /^[a-z]\d+$/i.test(last);

  if (pageSuffix && parts.length >= 3) {
    return `${parts[0]}-${parts[1]}-${last}`.toLowerCase();
  }

  if (parts.length === 2) {
    return `${parts[0]}-${parts[1]}`.toLowerCase();
  }

  return `${parts[0]}-${parts[1]}-…`.toLowerCase();
}

/** Short display label; full title available via formatPageTitle. */
export function displayPageName(stem: string, maxLength = 28): string {
  const slug = pageSlug(stem);
  if (slug.length <= maxLength) return slug;
  return `${slug.slice(0, maxLength - 1)}…`;
}
