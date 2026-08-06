import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  AGENT_INSTALL_COMMAND,
  AGENT_PACKAGE_NAME,
  AGENT_PAIR_COMMAND,
  AGENT_RUN_COMMAND,
} from "../../inference/constants";
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
  it("reports capacity and offers no host picker of its own", () => {
    renderBanner();

    // The one control that changes the account setting lives in editor
    // settings. A second copy here would read as a per-run choice.
    expect(screen.queryByRole("checkbox")).toBeNull();
    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByText(/local only/i)).toBeNull();
    expect(screen.queryByText(/nothing is sent to the cloud/i)).toBeNull();
  });

  it("offers a retry instead of polling in the background", () => {
    const { props } = renderBanner();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(props.onRetry).toHaveBeenCalledTimes(1);
  });

  it("hides the agent controls once the account prefers the cloud", () => {
    renderBanner({ preferLocalInference: false });

    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /how to run it here/i }),
    ).toBeNull();
  });

  it("says an absent agent sends jobs to the cloud rather than failing them", () => {
    renderBanner({ preferLocalInference: true });

    expect(screen.getByText(/so jobs go to the cloud/i)).toBeTruthy();
    expect(screen.queryByText(/runs will fail/i)).toBeNull();
  });

  it("shows the compact banner, not a blocking modal, when the agent is absent", () => {
    renderBanner();

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(
      screen.getByRole("button", { name: /how to run it here/i }),
    ).toBeTruthy();
  });

  it("teaches the install as commands, never as a download link", () => {
    renderBanner();
    openInstructions();

    const dialog = screen.getByRole("dialog", {
      name: /run inference on this computer/i,
    });
    expect(dialog.textContent).toContain(AGENT_INSTALL_COMMAND);
    expect(dialog.textContent).toContain(AGENT_PAIR_COMMAND);
    expect(dialog.textContent).toContain(AGENT_RUN_COMMAND);
    expect(AGENT_INSTALL_COMMAND).toContain(AGENT_PACKAGE_NAME);

    // The four per-OS installer URLs are gone with the workflow that built
    // them (#61). A link here would 404 at the next release cut; a command
    // cannot. Nothing in this panel may point at a release asset again.
    expect(screen.queryAllByRole("link")).toEqual([]);
    expect(dialog.innerHTML).not.toContain("releases/latest");
    expect(dialog.innerHTML).not.toContain("github.com");
  });

  it("gives one set of instructions rather than a platform picker", () => {
    vi.stubGlobal("navigator", {
      platform: "Win32",
      userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    });

    renderBanner();
    openInstructions();

    // One **published package**, so there is nothing to choose between.
    expect(
      screen.queryByRole("button", { name: /other platforms/i }),
    ).toBeNull();
    expect(screen.queryByText(/download for/i)).toBeNull();
    expect(screen.getByText(/macOS, Windows and Linux/i)).toBeTruthy();

    vi.unstubAllGlobals();
  });

  it("calls onUseCloudInstead from the modal", () => {
    const { props } = renderBanner();
    openInstructions();

    fireEvent.click(
      screen.getByRole("button", { name: /use cloud inference instead/i }),
    );
    expect(props.onUseCloudInstead).toHaveBeenCalledTimes(1);
  });

  it("returns to the compact banner after dismissing the modal", () => {
    renderBanner();
    openInstructions();

    fireEvent.click(screen.getByRole("button", { name: /not now/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(
      screen.getByRole("button", { name: /how to run it here/i }),
    ).toBeTruthy();
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
