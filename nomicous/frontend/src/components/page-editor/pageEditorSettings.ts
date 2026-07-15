const STORAGE_KEY = "nomicous_page_editor_settings";

export type PageEditorCanvasSettings = {
  /** Multiplier for segment/block overlay stroke width (0.5-4). */
  overlayStrokeWidth: number;
  /** Multiplier for Kraken/layout baseline stroke width (0.25-2.5). */
  baselineStrokeWidth: number;
  /** Segment polygon fill strength (0-0.35). */
  segmentFillOpacity: number;
  /** Multiplier for polygon corner handles (0.4-2.5). */
  handleSize: number;
  showLayoutBlocks: boolean;
  showBaselines: boolean;
};

export const DEFAULT_PAGE_EDITOR_SETTINGS: PageEditorCanvasSettings = {
  overlayStrokeWidth: 1.25,
  baselineStrokeWidth: 0.75,
  segmentFillOpacity: 0.1,
  handleSize: 0.75,
  showLayoutBlocks: true,
  showBaselines: false,
};

function clampNumber(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  return typeof value === "number" && value >= min && value <= max
    ? value
    : fallback;
}

export function loadPageEditorSettings(): PageEditorCanvasSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PAGE_EDITOR_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<PageEditorCanvasSettings>;
    return {
      overlayStrokeWidth: clampNumber(
        parsed.overlayStrokeWidth,
        0.5,
        4,
        DEFAULT_PAGE_EDITOR_SETTINGS.overlayStrokeWidth,
      ),
      baselineStrokeWidth: clampNumber(
        parsed.baselineStrokeWidth,
        0.25,
        2.5,
        DEFAULT_PAGE_EDITOR_SETTINGS.baselineStrokeWidth,
      ),
      segmentFillOpacity: clampNumber(
        parsed.segmentFillOpacity,
        0,
        0.35,
        DEFAULT_PAGE_EDITOR_SETTINGS.segmentFillOpacity,
      ),
      handleSize: clampNumber(
        parsed.handleSize,
        0.4,
        2.5,
        DEFAULT_PAGE_EDITOR_SETTINGS.handleSize,
      ),
      showLayoutBlocks:
        typeof parsed.showLayoutBlocks === "boolean"
          ? parsed.showLayoutBlocks
          : DEFAULT_PAGE_EDITOR_SETTINGS.showLayoutBlocks,
      showBaselines:
        typeof parsed.showBaselines === "boolean"
          ? parsed.showBaselines
          : DEFAULT_PAGE_EDITOR_SETTINGS.showBaselines,
    };
  } catch {
    return DEFAULT_PAGE_EDITOR_SETTINGS;
  }
}

export function savePageEditorSettings(
  settings: PageEditorCanvasSettings,
): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}
