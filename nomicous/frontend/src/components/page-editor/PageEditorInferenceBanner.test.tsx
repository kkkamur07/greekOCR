import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  INFERENCE_HELPER_LINUX_TARBALL_URL,
  INFERENCE_HELPER_MACOS_INTEL_DMG_URL,
  INFERENCE_HELPER_MACOS_DMG_URL,
  INFERENCE_HELPER_RELEASES_URL,
  INFERENCE_HELPER_WINDOWS_ZIP_URL,
} from "../../inference/constants";
import type { InferenceRouting } from "../../inference/preference";
import { PageEditorInferenceBanner } from "./PageEditorInferenceBanner";

type BannerProps = {
  helperAvailable: boolean;
  probing: boolean;
  routing: InferenceRouting;
  onRoutingChange: (routing: InferenceRouting) => void;
  onRetry: () => void;
  onUseCloudInstead: () => void;
};

function renderBanner(overrides: Partial<BannerProps> = {}) {
  const props: BannerProps = {
    helperAvailable: false,
    probing: false,
    routing: "auto",
    onRoutingChange: vi.fn(),
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

  it("always offers the three routing choices in plain language", () => {
    renderBanner();

    expect(screen.getByRole("radio", { name: "Automatic" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Local only" })).toBeTruthy();
    expect(screen.getByRole("radio", { name: "Cloud only" })).toBeTruthy();
  });

  it("reports the chosen routing", () => {
    const { props } = renderBanner();

    fireEvent.click(screen.getByRole("radio", { name: "Local only" }));

    expect(props.onRoutingChange).toHaveBeenCalledWith("local-only");
  });

  it("offers a retry instead of polling in the background", () => {
    const { props } = renderBanner();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(props.onRetry).toHaveBeenCalledTimes(1);
  });

  it("hides helper controls once cloud-only is selected", () => {
    renderBanner({ routing: "cloud-only" });

    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /install helper/i }),
    ).toBeNull();
  });

  it("warns that runs will fail under local-only without a helper", () => {
    renderBanner({ routing: "local-only" });

    expect(screen.getByText(/runs will fail until you start it/i)).toBeTruthy();
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
        routing="auto"
        onRoutingChange={vi.fn()}
        onRetry={vi.fn()}
        onUseCloudInstead={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /install helper/i }),
    ).toBeNull();
    expect(screen.getByText(/helper found on this computer/i)).toBeTruthy();
  });
});
