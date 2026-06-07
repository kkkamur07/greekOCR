import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PageCard from "./PageCard";
import type { PageSummary } from "@/types/api";

const BASE_PAGE: PageSummary = {
  stem: "folio",
  has_transcription: true,
  segment_count: 2,
  export_dirty: false,
  locked: false,
  pairing: {
    paired_count: 2,
    unpaired_count: 0,
    text_line_count: 2,
    unused_line_count: 0,
  },
};

describe("PageCard", () => {
  it("shows locked badge when page is locked", () => {
    render(<PageCard page={{ ...BASE_PAGE, locked: true }} />);
    expect(screen.getByText("locked")).toBeInTheDocument();
  });

  it("hides locked badge when page is unlocked", () => {
    render(<PageCard page={BASE_PAGE} />);
    expect(screen.queryByText("locked")).not.toBeInTheDocument();
  });
});
