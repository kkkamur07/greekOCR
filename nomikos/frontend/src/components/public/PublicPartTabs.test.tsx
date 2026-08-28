import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PublicPartTabs } from "./PublicPartTabs";

function makeParts(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: `part-${i + 1}`,
    label: `Page ${i + 1}`,
  }));
}

describe("PublicPartTabs", () => {
  it("renders a tab for every part and no +N overflow indicator", () => {
    const parts = makeParts(18);
    render(
      <PublicPartTabs parts={parts} activeId="part-1" onChange={vi.fn()} />,
    );

    expect(screen.getAllByRole("tab")).toHaveLength(18);
    expect(screen.getByRole("tab", { name: "Page 18" })).toBeInTheDocument();
    expect(screen.queryByText(/^\+/)).not.toBeInTheDocument();
  });

  it("wraps ArrowRight from the last tab back to the first across all parts", () => {
    const parts = makeParts(18);
    const onChange = vi.fn();
    render(
      <PublicPartTabs parts={parts} activeId="part-18" onChange={onChange} />,
    );

    const lastTab = screen.getByRole("tab", { name: "Page 18" });
    fireEvent.keyDown(lastTab, { key: "ArrowRight" });

    expect(onChange).toHaveBeenCalledWith("part-1");
  });

  it("reaches page 18 with End from the first tab", () => {
    const parts = makeParts(18);
    const onChange = vi.fn();
    render(
      <PublicPartTabs parts={parts} activeId="part-1" onChange={onChange} />,
    );

    const firstTab = screen.getByRole("tab", { name: "Page 1" });
    fireEvent.keyDown(firstTab, { key: "End" });

    expect(onChange).toHaveBeenCalledWith("part-18");
  });
});
