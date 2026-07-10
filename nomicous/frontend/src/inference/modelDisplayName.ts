/**
 * Human-friendly label for a registry model id, used in user-facing banners.
 * Avoids leaking raw registry ids like `greek-kraken-segment-v1` into the UI.
 */
export function modelDisplayName(registryModelId: string | null): string {
  if (!registryModelId) return "inference model";
  const id = registryModelId.toLowerCase();
  if (id.includes("kraken") || id.includes("segment")) {
    return "kraken segmentation model";
  }
  if (id.includes("calamari") || id.includes("htr") || id.includes("ocr")) {
    return "OCR model";
  }
  return "inference model";
}
