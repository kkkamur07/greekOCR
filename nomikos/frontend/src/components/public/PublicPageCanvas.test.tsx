import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { Region } from "../../types";
import { PublicPageCanvas } from "./PublicPageCanvas";

vi.mock("./PublicZoomSurface", () => ({
  PublicZoomSurface: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

const REGIONS: Region[] = [
  {
    id: 1,
    boundary: [
      [10, 10],
      [50, 10],
      [50, 30],
      [10, 30],
    ],
    bbox: [10, 10, 50, 30],
  },
];

function stubNaturalSize(img: HTMLImageElement, width: number, height: number) {
  // jsdom never lays out elements, so clientWidth/clientHeight default to 0.
  // Stubbing all four keeps both the display size and the coordinate size
  // real, the way a decoded image in a real browser would report them.
  Object.defineProperty(img, "naturalWidth", {
    value: width,
    configurable: true,
  });
  Object.defineProperty(img, "naturalHeight", {
    value: height,
    configurable: true,
  });
  Object.defineProperty(img, "clientWidth", {
    value: width,
    configurable: true,
  });
  Object.defineProperty(img, "clientHeight", {
    value: height,
    configurable: true,
  });
}

/**
 * A box before a decode, the way a real browser gives one.
 *
 * The `<img>` is laid out from the page's aspect ratio the moment it is in the
 * DOM, so its ResizeObserver fires with a real size while the bytes are still
 * arriving. jsdom lays nothing out and never fires one, which is why the other
 * tests here reach `imageLoaded` and `displaySize` in the same step and cannot
 * tell which of the two is holding the overlay back. This produces the state
 * where they disagree.
 */
function withUndecodedLayoutBox(width: number, height: number) {
  const observers = globalThis.ResizeObserver;
  const proto = globalThis.HTMLElement.prototype;
  const widthDescriptor = Object.getOwnPropertyDescriptor(proto, "clientWidth");
  const heightDescriptor = Object.getOwnPropertyDescriptor(
    proto,
    "clientHeight",
  );

  Object.defineProperty(proto, "clientWidth", {
    value: width,
    configurable: true,
  });
  Object.defineProperty(proto, "clientHeight", {
    value: height,
    configurable: true,
  });
  globalThis.ResizeObserver = class {
    constructor(private readonly callback: ResizeObserverCallback) {}
    observe() {
      this.callback([], this as unknown as ResizeObserver);
    }
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;

  return () => {
    globalThis.ResizeObserver = observers;
    if (widthDescriptor)
      Object.defineProperty(proto, "clientWidth", widthDescriptor);
    if (heightDescriptor)
      Object.defineProperty(proto, "clientHeight", heightDescriptor);
  };
}

describe("PublicPageCanvas", () => {
  it("keeps the overlay off an image that has a box but no pixels yet", () => {
    const restore = withUndecodedLayoutBox(640, 900);
    try {
      render(
        <PublicPageCanvas
          imageUrl="/page-1.jpg"
          layoutWidth={640}
          layoutHeight={900}
          regions={REGIONS}
          selectedRegionId={null}
          onSelectRegion={vi.fn()}
        />,
      );

      // This is the whole bug: `displaySize` is already real and `coordSize`
      // already holds the layout dimensions from the props, so every condition
      // except "has the image actually loaded" is satisfied. Drawing here puts
      // polygons over blank space, and over the wrong box the moment the real
      // image reports a natural size that differs from the layout's.
      expect(
        screen.queryByRole("button", { name: "Line 1" }),
      ).not.toBeInTheDocument();

      const img = screen.getByAltText("Manuscript page") as HTMLImageElement;
      stubNaturalSize(img, 1280, 1800);
      fireEvent.load(img);

      const line = screen.getByRole("button", { name: "Line 1" });
      // And it is drawn against the decoded image's own coordinates, not the
      // layout figures it was holding a moment earlier.
      expect(line.closest("svg")?.getAttribute("viewBox")).toBe(
        "0 0 1280 1800",
      );
    } finally {
      restore();
    }
  });

  it("does not render polygons before the image loads, and does after", () => {
    render(
      <PublicPageCanvas
        imageUrl="/page-1.jpg"
        layoutWidth={640}
        layoutHeight={900}
        regions={REGIONS}
        selectedRegionId={null}
        onSelectRegion={vi.fn()}
      />,
    );

    const img = screen.getByAltText("Manuscript page") as HTMLImageElement;
    expect(
      screen.queryByRole("button", { name: "Line 1" }),
    ).not.toBeInTheDocument();

    stubNaturalSize(img, 640, 900);
    fireEvent.load(img);

    expect(screen.getByRole("button", { name: "Line 1" })).toBeInTheDocument();
  });

  it("clears the overlay when imageUrl changes until the new image loads", () => {
    const { rerender } = render(
      <PublicPageCanvas
        imageUrl="/page-1.jpg"
        layoutWidth={640}
        layoutHeight={900}
        regions={REGIONS}
        selectedRegionId={null}
        onSelectRegion={vi.fn()}
      />,
    );

    let img = screen.getByAltText("Manuscript page") as HTMLImageElement;
    stubNaturalSize(img, 640, 900);
    fireEvent.load(img);
    expect(screen.getByRole("button", { name: "Line 1" })).toBeInTheDocument();

    rerender(
      <PublicPageCanvas
        imageUrl="/page-2.jpg"
        layoutWidth={640}
        layoutHeight={900}
        regions={REGIONS}
        selectedRegionId={null}
        onSelectRegion={vi.fn()}
      />,
    );

    // The previous page's overlay must not linger over the not-yet-loaded
    // next image.
    expect(
      screen.queryByRole("button", { name: "Line 1" }),
    ).not.toBeInTheDocument();

    img = screen.getByAltText("Manuscript page") as HTMLImageElement;
    stubNaturalSize(img, 640, 900);
    fireEvent.load(img);

    expect(screen.getByRole("button", { name: "Line 1" })).toBeInTheDocument();
  });

  it("shows a plain failure message when the image fails to load", () => {
    render(
      <PublicPageCanvas
        imageUrl="/broken.jpg"
        layoutWidth={640}
        layoutHeight={900}
        regions={REGIONS}
        selectedRegionId={null}
        onSelectRegion={vi.fn()}
      />,
    );

    const img = screen.getByAltText("Manuscript page");
    fireEvent.error(img);

    expect(
      screen.getByText("This page image could not be loaded."),
    ).toBeInTheDocument();
    expect(screen.queryByAltText("Manuscript page")).not.toBeInTheDocument();
  });
});
