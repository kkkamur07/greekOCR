import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PublicPageCanvas } from "./PublicPageCanvas";

vi.mock("./PublicZoomSurface", () => ({
  PublicZoomSurface: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));

describe("PublicPageCanvas", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("selects and clears a line geometry with the keyboard", () => {
    vi.spyOn(HTMLImageElement.prototype, "clientWidth", "get").mockReturnValue(
      640,
    );
    vi.spyOn(HTMLImageElement.prototype, "clientHeight", "get").mockReturnValue(
      900,
    );
    const onSelectRegion = vi.fn();
    render(
      <PublicPageCanvas
        imageUrl="/page.webp"
        layoutWidth={640}
        layoutHeight={900}
        regions={[
          {
            id: 1,
            bbox: [10, 10, 50, 30],
            boundary: [
              [10, 10],
              [50, 10],
              [50, 30],
              [10, 30],
            ],
          },
        ]}
        selectedRegionId={null}
        onSelectRegion={onSelectRegion}
      />,
    );

    const image = screen.getByAltText("Manuscript page");
    fireEvent.load(image);
    const line = screen.getByRole("button", { name: "Line 1" });

    fireEvent.keyDown(line, { key: "Enter" });
    fireEvent.keyDown(line, { key: " " });

    expect(onSelectRegion).toHaveBeenNthCalledWith(1, 1);
    expect(onSelectRegion).toHaveBeenNthCalledWith(2, 1);
  });
});
