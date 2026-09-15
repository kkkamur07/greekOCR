import { act, render } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ZOOM_SETTLE_MS } from "../../utils/zoomLayerHint";

type TransformedRef = { state: { scale: number } };

type StubWrapperProps = {
  children: ReactNode | (() => ReactNode);
  onTransformed?: (ref: TransformedRef) => void;
};

/**
 * The real pan/zoom library needs layout, which jsdom never does, so the other
 * suites here replace this surface wholesale. Stubbing the library instead
 * keeps the surface itself under test and hands back its `onTransformed` so a
 * gesture can be played through it.
 */
let onTransformed: ((ref: TransformedRef) => void) | undefined;

vi.mock("react-zoom-pan-pinch", () => ({
  TransformWrapper: (props: StubWrapperProps) => {
    onTransformed = props.onTransformed;
    return (
      <>
        {typeof props.children === "function"
          ? props.children()
          : props.children}
      </>
    );
  },
  TransformComponent: ({ children }: { children: ReactNode }) => (
    <>{children}</>
  ),
}));

function surface(): HTMLElement {
  const node = document.querySelector<HTMLElement>(".pub-zoom-surface");
  if (!node) throw new Error("zoom surface not rendered");
  return node;
}

describe("PublicZoomSurface compositor hint", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    onTransformed = undefined;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("promotes the layer while the view moves and releases it once it settles", async () => {
    const { PublicZoomSurface } = await import("./PublicZoomSurface");
    render(<PublicZoomSurface>page</PublicZoomSurface>);

    expect(surface().className).not.toContain("pub-zoom-surface--transforming");

    act(() => onTransformed?.({ state: { scale: 2 } }));
    expect(surface().className).toContain("pub-zoom-surface--transforming");

    act(() => {
      vi.advanceTimersByTime(ZOOM_SETTLE_MS - 1);
    });
    expect(surface().className).toContain("pub-zoom-surface--transforming");

    // Once the transform stops arriving the hint goes away, which is what lets
    // the browser re-raster the overlay crisply at the new scale.
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(surface().className).not.toContain("pub-zoom-surface--transforming");
  });
});
