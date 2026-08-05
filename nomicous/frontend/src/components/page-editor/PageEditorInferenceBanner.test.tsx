import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  INFERENCE_HELPER_LINUX_TARBALL_URL,
  INFERENCE_HELPER_MACOS_INTEL_DMG_URL,
  INFERENCE_HELPER_MACOS_DMG_URL,
  INFERENCE_HELPER_RELEASES_URL,
  INFERENCE_HELPER_WINDOWS_ZIP_URL,
} from "../../inference/constants";
import { PageEditorInferenceBanner } from "./PageEditorInferenceBanner";

type BannerProps = {
  helperAvailable: boolean;
  probing: boolean;
  preferLocalInference: boolean;
  onRetry: () => void;
  onUseCloudInstead: () => void;
};

function renderBanner(overrides: Partial<BannerProps> = {}) {
  const props: BannerProps = {
    helperAvailable: false,
    probing: false,
    preferLocalInference: true,
    onRetry: vi.fn(),
    onUseCloudInstead: vi.fn(),
    ...overrides,
  };
  return { props, ...render(<PageEditorInferenceBanner {...props} />) };
}

describe("PageEditorInferenceBanner", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

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

  it("hides helper controls once the account prefers the cloud", () => {
    renderBanner({ preferLocalInference: false });

    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /install helper/i }),
    ).toBeNull();
  });

  it("says an absent agent sends jobs to the cloud rather than failing them", () => {
    renderBanner({ preferLocalInference: true });

    expect(screen.getByText(/so jobs go to the cloud/i)).toBeTruthy();
    expect(screen.queryByText(/runs will fail/i)).toBeNull();
  });

  it("shows the compact banner (not a blocking modal) when helper is unavailable", () => {
    renderBanner();

    expect(
      screen.queryByRole("dialog", { name: /install inference helper/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /install helper/i }),
    ).toBeTruthy();
  });

  it("opens the install modal only after clicking install helper", () => {
    renderBanner();

    fireEvent.click(screen.getByRole("button", { name: /install helper/i }));
    expect(
      screen.getByRole("dialog", { name: /install inference helper/i }),
    ).toBeTruthy();
    expect(screen.getByText(/detects the helper automatically/i)).toBeTruthy();
  });

  it("shows a single primary download for the detected OS", () => {
    vi.stubGlobal("navigator", {
      platform: "Win32",
      userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    });

    renderBanner();

    fireEvent.click(screen.getByRole("button", { name: /install helper/i }));
    const primary = screen.getByRole("link", {
      name: /download for this pc \(windows\)/i,
    });
    expect(primary).toHaveAttribute("href", INFERENCE_HELPER_WINDOWS_ZIP_URL);
    expect(primary.className).toContain("btn-primary");
    expect(
      screen.queryByRole("link", { name: /download for macos/i }),
    ).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /other platforms/i }));
    expect(
      screen.getByRole("link", {
        name: /download for macos \(apple silicon\)/i,
      }),
    ).toHaveAttribute("href", INFERENCE_HELPER_MACOS_DMG_URL);
    expect(
      screen.getByRole("link", { name: /download for macos \(intel\)/i }),
    ).toHaveAttribute("href", INFERENCE_HELPER_MACOS_INTEL_DMG_URL);
    expect(
      screen.getByRole("link", { name: /view release notes/i }),
    ).toHaveAttribute("href", INFERENCE_HELPER_RELEASES_URL);
    expect(INFERENCE_HELPER_RELEASES_URL).toContain("/releases/latest");
    expect(INFERENCE_HELPER_LINUX_TARBALL_URL).toContain(
      "/releases/latest/download/",
    );
  });

  it("calls onUseCloudInstead from the modal", () => {
    const { props } = renderBanner();

    fireEvent.click(screen.getByRole("button", { name: /install helper/i }));
    fireEvent.click(
      screen.getByRole("button", { name: /use cloud inference instead/i }),
    );
    expect(props.onUseCloudInstead).toHaveBeenCalledTimes(1);
  });

  it("returns to the compact banner after dismissing the modal", () => {
    renderBanner();

    fireEvent.click(screen.getByRole("button", { name: /install helper/i }));
    fireEvent.click(screen.getByRole("button", { name: /not now/i }));
    expect(
      screen.queryByRole("dialog", { name: /install inference helper/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /install helper/i }),
    ).toBeTruthy();
  });

  it("does not nag while probing or when the helper is available", () => {
    const { rerender } = renderBanner({ probing: true });
    expect(
      screen.queryByRole("button", { name: /install helper/i }),
    ).toBeNull();

    rerender(
      <PageEditorInferenceBanner
        helperAvailable={true}
        probing={false}
        preferLocalInference={true}
        onRetry={vi.fn()}
        onUseCloudInstead={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /install helper/i }),
    ).toBeNull();
    expect(
      screen.getByText(/the agent is running on this computer/i),
    ).toBeTruthy();
  });
});
