import { beforeEach, describe, expect, it } from "vitest";

import {
  BASE_WHEEL_SMOOTH_STEP,
  BASE_WHEEL_STEP,
  DEFAULT_PAGE_EDITOR_SETTINGS,
  loadPageEditorSettings,
  savePageEditorSettings,
  wheelZoomConfig,
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
      wheelZoomSpeed: 1.5,
    });
    expect(loadPageEditorSettings().wheelZoomSpeed).toBe(1.5);
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

  it.each([0, 3, "1"])(
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
});
