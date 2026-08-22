import { beforeEach, describe, expect, it } from "vitest";

import {
  BASE_WHEEL_SMOOTH_STEP,
  BASE_WHEEL_STEP,
  DEFAULT_PAGE_EDITOR_SETTINGS,
  loadPageEditorSettings,
  savePageEditorSettings,
  wheelZoomConfig,
  wheelZoomPercentPerNotch,
} from "./pageEditorSettings";

describe("pageEditorSettings", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("falls back to the defaults when nothing is stored", () => {
    expect(loadPageEditorSettings()).toEqual(DEFAULT_PAGE_EDITOR_SETTINGS);
  });

  it("round-trips a saved wheel zoom speed", () => {
    savePageEditorSettings({
      ...DEFAULT_PAGE_EDITOR_SETTINGS,
      wheelZoomSpeed: 3,
    });
    expect(loadPageEditorSettings().wheelZoomSpeed).toBe(3);
  });

  it("keeps settings saved before the wheel zoom speed existed", () => {
    // A profile written by an older build has no wheelZoomSpeed key at all.
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ overlayStrokeWidth: 2, showBaselines: true }),
    );
    const loaded = loadPageEditorSettings();
    expect(loaded.overlayStrokeWidth).toBe(2);
    expect(loaded.showBaselines).toBe(true);
    expect(loaded.wheelZoomSpeed).toBe(
      DEFAULT_PAGE_EDITOR_SETTINGS.wheelZoomSpeed,
    );
  });

  it.each([0, 0.1, 7, -1, "3", null])(
    "ignores an out-of-range or non-numeric wheel zoom speed (%j)",
    (value) => {
      localStorage.setItem(
        "nomikos_page_editor_settings",
        JSON.stringify({ wheelZoomSpeed: value }),
      );
      expect(loadPageEditorSettings().wheelZoomSpeed).toBe(
        DEFAULT_PAGE_EDITOR_SETTINGS.wheelZoomSpeed,
      );
    },
  );

  it("scales both wheel steps together", () => {
    expect(wheelZoomConfig(1)).toEqual({
      step: BASE_WHEEL_STEP,
      smoothStep: BASE_WHEEL_SMOOTH_STEP,
    });
    expect(wheelZoomConfig(2.5)).toEqual({
      step: BASE_WHEEL_STEP * 2.5,
      smoothStep: BASE_WHEEL_SMOOTH_STEP * 2.5,
    });
  });

  it("describes one mouse-wheel notch in percent", () => {
    // 0.0006 * 120 = 7.2% at the default speed: the value the PR 85 fix was tuned to.
    expect(wheelZoomPercentPerNotch(1)).toBe(7);
    expect(wheelZoomPercentPerNotch(4)).toBe(29);
  });
});
