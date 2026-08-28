import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_PAGE_EDITOR_SETTINGS,
  loadPageEditorSettings,
  savePageEditorSettings,
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

  it.each([0, 4, "1"])(
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
});
