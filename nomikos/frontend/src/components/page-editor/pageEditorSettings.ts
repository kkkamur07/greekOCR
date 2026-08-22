const STORAGE_KEY = "nomikos_page_editor_settings";

/**
 * react-zoom-pan-pinch adds `smoothStep * |deltaY|` to the scale per wheel
 * event. 0.006 is the value the editor shipped with; it was cut tenfold on
 * 2026-08-19 to fix a "page flies off" symptom that was really a stale
 * pan-velocity bug (see utils/zoomPanVelocity.ts), which left trackpad zooming
 * crawling. The original is the 1× default again; `wheelZoomSpeed` scales it,
 * and 0.1× is the tamed value for a mouse wheel that reports big deltas.
 */
export const BASE_WHEEL_SMOOTH_STEP = 0.006;
export const BASE_WHEEL_STEP = 0.06;
export const WHEEL_ZOOM_SPEED_MIN = 0.1;
export const WHEEL_ZOOM_SPEED_MAX = 2;

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
  /** Multiplier on how far one wheel notch or trackpad step zooms (0.1-2). */
  wheelZoomSpeed: number;
};

export const DEFAULT_PAGE_EDITOR_SETTINGS: PageEditorCanvasSettings = {
  overlayStrokeWidth: 1.25,
  baselineStrokeWidth: 0.75,
  segmentFillOpacity: 0.1,
  handleSize: 0.75,
  showLayoutBlocks: true,
  showBaselines: false,
  wheelZoomSpeed: 1,
};

/** The `wheel` config for the editor's TransformWrapper at a given speed. */
export function wheelZoomConfig(wheelZoomSpeed: number): {
  step: number;
  smoothStep: number;
} {
  return {
    step: BASE_WHEEL_STEP * wheelZoomSpeed,
    smoothStep: BASE_WHEEL_SMOOTH_STEP * wheelZoomSpeed,
  };
}

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
      wheelZoomSpeed: clampNumber(
        parsed.wheelZoomSpeed,
        WHEEL_ZOOM_SPEED_MIN,
        WHEEL_ZOOM_SPEED_MAX,
        DEFAULT_PAGE_EDITOR_SETTINGS.wheelZoomSpeed,
      ),
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
