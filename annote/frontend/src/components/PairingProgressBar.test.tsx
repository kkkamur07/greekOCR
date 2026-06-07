import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PairingProgressBar from "./PairingProgressBar";

describe("PairingProgressBar", () => {
  it("shows a complete badge when pairing is finished", () => {
    render(
      <PairingProgressBar
        progress={{
          paired_count: 2,
          unpaired_count: 0,
          text_line_count: 2,
          unused_line_count: 0,
        }}
      />,
    );

    expect(screen.getByText("complete")).toBeInTheDocument();
  });

  it("does not show a complete badge while work remains", () => {
    render(
      <PairingProgressBar
        progress={{
          paired_count: 1,
          unpaired_count: 1,
          text_line_count: 2,
          unused_line_count: 1,
        }}
      />,
    );

    expect(screen.queryByText("complete")).not.toBeInTheDocument();
  });
});
