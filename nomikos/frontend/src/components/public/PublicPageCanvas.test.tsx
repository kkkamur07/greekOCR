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

describe("PublicPageCanvas", () => {
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
