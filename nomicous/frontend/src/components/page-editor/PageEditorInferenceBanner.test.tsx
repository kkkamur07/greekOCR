import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PageEditorInferenceBanner } from "./PageEditorInferenceBanner";

type BannerProps = {
  hasLocalCapacity: boolean;
  loading: boolean;
  preferLocalInference: boolean;
  onRetry: () => void;
  onUseCloudInstead: () => void;
};

function renderBanner(overrides: Partial<BannerProps> = {}) {
  const props: BannerProps = {
    hasLocalCapacity: false,
    loading: false,
    preferLocalInference: true,
    onRetry: vi.fn(),
    onUseCloudInstead: vi.fn(),
    ...overrides,
  };
  return { props, ...render(<PageEditorInferenceBanner {...props} />) };
}

function openInstructions() {
  fireEvent.click(screen.getByRole("button", { name: /how to run it here/i }));
}

describe("PageEditorInferenceBanner", () => {
  it("hides the agent controls once the account prefers the cloud", () => {
    renderBanner({ preferLocalInference: false });

    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /how to run it here/i }),
    ).toBeNull();
  });

  it("teaches the install as commands, never as a download link", () => {
    renderBanner();
    openInstructions();

    const dialog = screen.getByRole("dialog", {
      name: /run inference on this computer/i,
    });

    // The four per-OS installer URLs are gone with the workflow that built
    // them (#61). A link here would 404 at the next release cut; a command
    // cannot. Nothing in this panel may point at a release asset again.
    expect(screen.queryAllByRole("link")).toEqual([]);
    expect(dialog.innerHTML).not.toContain("releases/latest");
    expect(dialog.innerHTML).not.toContain("github.com");
  });

  it("does not nag while reading the account or when the agent is running", () => {
    const { rerender } = renderBanner({ loading: true });
    expect(
      screen.queryByRole("button", { name: /how to run it here/i }),
    ).toBeNull();

    rerender(
      <PageEditorInferenceBanner
        hasLocalCapacity={true}
        loading={false}
        preferLocalInference={true}
        onRetry={vi.fn()}
        onUseCloudInstead={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /how to run it here/i }),
    ).toBeNull();
    expect(
      screen.getByText(/the agent is running on this computer/i),
    ).toBeTruthy();
  });
});
