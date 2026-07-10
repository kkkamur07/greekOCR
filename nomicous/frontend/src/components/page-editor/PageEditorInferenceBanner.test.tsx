import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PageEditorInferenceBanner } from "./PageEditorInferenceBanner";

describe("PageEditorInferenceBanner", () => {
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
    expect(
      screen.getByRole("link", { name: /download for macos/i }),
    ).toBeTruthy();
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
