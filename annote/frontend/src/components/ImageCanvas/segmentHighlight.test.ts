import { describe, expect, it } from "vitest";

import { getSegmentHighlight } from "./segmentHighlight";

describe("getSegmentHighlight", () => {
  it("uses green for paired segments and amber for unpaired segments", () => {
    const unpaired = getSegmentHighlight({ selected: false, paired: false });
    const paired = getSegmentHighlight({ selected: false, paired: true });

    expect(paired.stroke).toBe("#16a34a");
    expect(paired.fill).toContain("34,197,94");
    expect(unpaired.stroke).toBe("#d97706");
    expect(unpaired.fill).toContain("245,158,11");
  });

  it("keeps the selection highlight when a paired segment is selected", () => {
    const selectedUnpaired = getSegmentHighlight({ selected: true, paired: false });
    const selectedPaired = getSegmentHighlight({ selected: true, paired: true });

    expect(selectedPaired).toEqual(selectedUnpaired);
  });
});
