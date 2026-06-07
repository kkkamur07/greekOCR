import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ExportPreviewModal from "./ExportPreviewModal";

describe("ExportPreviewModal", () => {
  it("loads the segment preview image from the API", () => {
    render(
      <ExportPreviewModal
        stem="folio"
        segmentId="seg-1"
        segmentNumber={3}
        onClose={() => {}}
      />,
    );

    const image = screen.getByRole("img", { name: "Export preview for segment 3" });
    expect(image.getAttribute("src")).toMatch(/\/pages\/folio\/segments\/seg-1\/preview\?t=\d+$/);
  });
});
