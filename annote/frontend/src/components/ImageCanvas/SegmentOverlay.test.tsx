import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Segment } from "@/types/api";

import SegmentOverlay from "./SegmentOverlay";

const noop = () => {};

function makeSegment(overrides: Partial<Segment> = {}): Segment {
  return {
    id: "seg-1",
    number: 1,
    kind: "polygon",
    points: [
      [10, 10],
      [50, 10],
      [50, 40],
      [10, 40],
    ],
    paired_text_line_index: null,
    ...overrides,
  };
}

const baseProps = {
  imageWidth: 100,
  imageHeight: 100,
  selectedId: null,
  draftPoints: [] as [number, number][],
  editMode: false,
  visible: true,
  interactive: true,
  zoomScale: 1,
  onSelect: noop,
  clientToImage: () => null,
  selectedVertexIndex: null,
  onSelectVertex: noop,
  onVertexDrag: noop,
  onInsertVertex: noop,
  onRemoveVertex: noop,
};

describe("SegmentOverlay", () => {
  it("renders a paired segment with the paired highlight colors", () => {
    const segment = makeSegment({ paired_text_line_index: 2 });

    const { container } = render(<SegmentOverlay {...baseProps} segments={[segment]} />);

    const path = container.querySelector('[data-segment-id="seg-1"]');
    expect(path).not.toBeNull();
    expect(path).toHaveAttribute("fill", "rgba(34,197,94,0.15)");
    expect(path).toHaveAttribute("stroke", "#16a34a");
  });

  it("treats inline text override as paired for highlight color", () => {
    const segment = makeSegment({ text_override: "typed line" });

    const { container } = render(<SegmentOverlay {...baseProps} segments={[segment]} />);

    const path = container.querySelector('[data-segment-id="seg-1"]');
    expect(path).toHaveAttribute("stroke", "#16a34a");
  });
});
