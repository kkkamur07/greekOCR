export interface SegmentHighlight {
  fill: string;
  stroke: string;
  label: string;
}

const PAIRED: SegmentHighlight = {
  fill: "rgba(34,197,94,0.15)",
  stroke: "#16a34a",
  label: "#15803d",
};

const UNPAIRED: SegmentHighlight = {
  fill: "rgba(245,158,11,0.15)",
  stroke: "#d97706",
  label: "#b45309",
};

const SELECTED: SegmentHighlight = {
  fill: "rgba(59,130,246,0.25)",
  stroke: "#2563eb",
  label: "#1d4ed8",
};

export function getSegmentHighlight({
  selected,
  paired,
}: {
  selected: boolean;
  paired: boolean;
}): SegmentHighlight {
  if (selected) return SELECTED;
  if (paired) return PAIRED;
  return UNPAIRED;
}
