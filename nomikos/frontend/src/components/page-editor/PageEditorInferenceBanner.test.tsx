import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("lets a reader dismiss the idle note and remembers it", async () => {
    const hint = /jobs run on this computer/i;
    const { unmount } = renderBanner({ preferLocalInference: false });

    expect(screen.getByText(hint)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /dismiss this note/i }));
    expect(screen.queryByText(hint)).toBeNull();

    // A note dismissed on one page that returns on the next has not been
    // dismissed in any sense the reader would recognise.
    unmount();
    renderBanner({ preferLocalInference: false });
    await waitFor(() => {
      expect(screen.queryByText(hint)).toBeNull();
    });
  });

  it("never offers to dismiss live agent status", () => {
    renderBanner({ preferLocalInference: true, hasLocalCapacity: true });

    // With the preference on, this line is where a researcher learns their
    // agent stopped. Hiding it would hide the reason their jobs went to the
    // cloud, so the control is not offered at all.
    expect(
      screen.queryByRole("button", { name: /dismiss this note/i }),
    ).toBeNull();
    expect(
      screen.getByText(/the agent is running on this computer/i),
    ).toBeInTheDocument();
  });

  it("keeps showing the idle note when storage cannot be read", async () => {
    const getItem = vi
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new Error("blocked");
      });
    try {
      renderBanner({ preferLocalInference: false });
      await waitFor(() => {
        expect(
          screen.getByText(/jobs run on this computer/i),
        ).toBeInTheDocument();
      });
    } finally {
      getItem.mockRestore();
    }
  });

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
