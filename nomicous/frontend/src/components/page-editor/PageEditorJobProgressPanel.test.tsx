import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PageEditorJobProgressPanel } from "./PageEditorJobProgressPanel";

describe("PageEditorJobProgressPanel", () => {
  it("shows queued and active jobs when expanded", () => {
    render(
      <PageEditorJobProgressPanel
        jobs={[
          {
            id: "job-1",
            label: "Segment 2",
            kind: "transcription-segment",
            status: "waiting",
            error: null,
            progressLabel: "Transcribing 0/1 segment",
            finishedAt: null,
          },
        ]}
        activeCount={1}
        expanded
        onExpandedChange={() => undefined}
        onDismissCompleted={() => undefined}
      />,
    );

    expect(
      screen.getByRole("dialog", { name: /background jobs/i }),
    ).toBeTruthy();
    expect(screen.getByText("Segment 2")).toBeTruthy();
    expect(screen.getByText("Transcribing 0/1 segment")).toBeTruthy();
  });
});
