import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";
import { PageEditorPageXmlButton } from "./PageEditorPageXmlButton";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getPageXml: vi.fn(),
    },
  };
});

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const mockedGetPageXml = api.getPageXml as ReturnType<typeof vi.fn>;

describe("PageEditorPageXmlButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetPageXml.mockResolvedValue(
      new Blob(["<PcGts/>"], { type: "application/xml" }),
    );
  });

  it("downloads the PAGE XML for the current part", async () => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:xml"),
      revokeObjectURL: vi.fn(),
    });
    const clicked: { href: string | null; download: string }[] = [];
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        clicked.push({
          href: this.getAttribute("href"),
          download: this.download,
        });
      });

    render(
      <PageEditorPageXmlButton
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="My_Doc_page_1.xml"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /download page xml/i }));

    await waitFor(() => {
      expect(mockedGetPageXml).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
      );
      expect(clicked).toEqual([
        { href: "blob:xml", download: "My_Doc_page_1.xml" },
      ]);
    });

    clickSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it("surfaces API errors as a toast and re-enables the button", async () => {
    mockedGetPageXml.mockRejectedValue(new ApiError("Forbidden", 403));

    render(
      <PageEditorPageXmlButton
        projectId="project-1"
        documentId="doc-1"
        partId="part-1"
        downloadFilename="My_Doc_page_1.xml"
      />,
    );

    const button = screen.getByRole("button", { name: /download page xml/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith("Forbidden");
      expect(button).toBeEnabled();
    });
  });
});
