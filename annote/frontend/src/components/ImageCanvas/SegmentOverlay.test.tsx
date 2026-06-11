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
    source: "manual",
    kraken_ceiling: null,
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
  showKrakenCeiling: false,
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

  it("renders kraken segments with a dashed outline style", () => {
    const segment = makeSegment({ source: "kraken" });

    const { container } = render(<SegmentOverlay {...baseProps} segments={[segment]} />);

    const path = container.querySelector('[data-segment-id="seg-1"]');
    expect(path?.getAttribute("stroke-dasharray")).not.toBeNull();
  });

  it("renders the Kraken ceiling overlay above the segment stroke", () => {
    const ceiling: [number, number][] = [
      [0, 0],
      [60, 0],
      [60, 50],
      [0, 50],
    ];
    const segment = makeSegment({
      source: "kraken",
      kraken_ceiling: ceiling,
      points: ceiling,
    });

    const { container } = render(
      <SegmentOverlay {...baseProps} segments={[segment]} selectedId="seg-1" showKrakenCeiling />,
    );

    const group = container.querySelector('[data-segment-id="seg-1"]')?.parentElement;
    const children = Array.from(group?.children ?? []);
    const segmentIndex = children.findIndex((el) => el.getAttribute("data-segment-hit") !== null);
    const ceilingIndex = children.findIndex((el) => el.getAttribute("data-kraken-ceiling") !== null);
    expect(ceilingIndex).toBeGreaterThan(segmentIndex);
    expect(container.querySelector('[data-kraken-ceiling]')).toHaveAttribute("stroke", "#c026d3");
  });

  it("shows the Kraken ceiling overlay only for the selected kraken segment when enabled", () => {
    const ceiling: [number, number][] = [
      [0, 0],
      [60, 0],
      [60, 50],
      [0, 50],
    ];
    const segment = makeSegment({
      source: "kraken",
      kraken_ceiling: ceiling,
      points: [
        [10, 10],
        [50, 10],
        [50, 40],
        [10, 40],
      ],
    });

    const { container } = render(
      <SegmentOverlay {...baseProps} segments={[segment]} selectedId="seg-1" showKrakenCeiling />,
    );

    expect(container.querySelector('[data-kraken-ceiling]')).not.toBeNull();
  });

  it("hides the Kraken ceiling overlay when the toggle is off", () => {
    const segment = makeSegment({
      source: "kraken",
      kraken_ceiling: [
        [0, 0],
        [60, 0],
        [60, 50],
        [0, 50],
      ],
    });

    const { container } = render(
      <SegmentOverlay {...baseProps} segments={[segment]} selectedId="seg-1" showKrakenCeiling={false} />,
    );

    expect(container.querySelector('[data-kraken-ceiling]')).toBeNull();
  });
});
