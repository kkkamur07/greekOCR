import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  INFERENCE_HELPER_LINUX_TARBALL_URL,
  INFERENCE_HELPER_MACOS_DMG_URL,
  INFERENCE_HELPER_RELEASES_URL,
  INFERENCE_HELPER_WINDOWS_ZIP_URL,
} from "../../inference/constants";
import { PageEditorInferenceBanner } from "./PageEditorInferenceBanner";

describe("PageEditorInferenceBanner", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the compact banner (not a blocking modal) when helper is unavailable", () => {
    render(
      <PageEditorInferenceBanner
        helperAvailable={false}
        probing={false}
        preferCloud={false}
        onUseCloudInstead={vi.fn()}
      />,
    );

    expect(
      screen.queryByRole("dialog", { name: /install inference helper/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /install helper/i }),
    ).toBeTruthy();
  });

  it("opens the install modal only after clicking install helper", () => {
    render(
      <PageEditorInferenceBanner
        helperAvailable={false}
        probing={false}
        preferCloud={false}
        onUseCloudInstead={vi.fn()}
      />,
    );

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

    render(
      <PageEditorInferenceBanner
        helperAvailable={false}
        probing={false}
        preferCloud={false}
        onUseCloudInstead={vi.fn()}
      />,
    );

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
      screen.getByRole("link", { name: /download for macos/i }),
    ).toHaveAttribute("href", INFERENCE_HELPER_MACOS_DMG_URL);
    expect(
      screen.getByRole("link", { name: /view release notes/i }),
    ).toHaveAttribute("href", INFERENCE_HELPER_RELEASES_URL);
    expect(INFERENCE_HELPER_RELEASES_URL).toContain("/releases/latest");
    expect(INFERENCE_HELPER_LINUX_TARBALL_URL).toContain(
      "/releases/latest/download/",
    );
  });

  it("calls onUseCloudInstead from the modal", () => {
    const onUseCloudInstead = vi.fn();
    render(
      <PageEditorInferenceBanner
        helperAvailable={false}
        probing={false}
        preferCloud={false}
        onUseCloudInstead={onUseCloudInstead}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /install helper/i }));
    fireEvent.click(
      screen.getByRole("button", { name: /use cloud inference instead/i }),
    );
    expect(onUseCloudInstead).toHaveBeenCalledTimes(1);
  });

  it("returns to the compact banner after dismissing the modal", () => {
    render(
      <PageEditorInferenceBanner
        helperAvailable={false}
        probing={false}
        preferCloud={false}
        onUseCloudInstead={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /install helper/i }));
    fireEvent.click(screen.getByRole("button", { name: /not now/i }));
    expect(
      screen.queryByRole("dialog", { name: /install inference helper/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /install helper/i }),
    ).toBeTruthy();
  });

  it("renders nothing while probing or when helper is available", () => {
    const { container, rerender } = render(
      <PageEditorInferenceBanner
        helperAvailable={false}
        probing={true}
        preferCloud={false}
        onUseCloudInstead={vi.fn()}
      />,
    );
    expect(container).toBeEmptyDOMElement();

    rerender(
      <PageEditorInferenceBanner
        helperAvailable={true}
        probing={false}
        preferCloud={false}
        onUseCloudInstead={vi.fn()}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
