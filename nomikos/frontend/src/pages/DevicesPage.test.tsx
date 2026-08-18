import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import { devicesApi, type DeviceResponse } from "../api/resources";
import * as session from "../auth/session";
import { DevicesPage } from "./DevicesPage";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, me: vi.fn() },
  };
});

vi.mock("../api/resources", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/resources")>();
  return {
    ...actual,
    devicesApi: {
      ...actual.devicesApi,
      listDevices: vi.fn(),
      revokeDevice: vi.fn(),
    },
  };
});

const online: DeviceResponse = {
  id: "9d5a9d54-0000-4000-8000-000000000010",
  name: "Ada's ThinkPad",
  platform: "linux-x86_64",
  helper_version: "0.4.1",
  status: "online",
  token_prefix: "nmd_7f3a",
  paired_at: "2026-08-01T09:00:00Z",
  last_seen_at: "2026-08-06T08:55:00Z",
  last_seen_ip: "10.0.0.4",
};

const revoked: DeviceResponse = {
  id: "9d5a9d54-0000-4000-8000-000000000011",
  name: "Old iMac",
  platform: "darwin-arm64",
  helper_version: "0.3.0",
  status: "revoked",
  token_prefix: "nmd_11bc",
  paired_at: "2026-05-01T09:00:00Z",
  revoked_at: "2026-07-02T12:00:00Z",
};

describe("DevicesPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
    vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
    vi.mocked(api.me).mockResolvedValue({
      id: "user-1",
      email: "ada@example.edu",
      username: "ada",
      created_at: "2026-01-01T00:00:00Z",
    });
    vi.mocked(devicesApi.listDevices).mockResolvedValue([online]);
    vi.mocked(devicesApi.revokeDevice).mockResolvedValue(undefined);
  });

  it("lists a device with its status, last seen and token prefix", async () => {
    render(<DevicesPage />);

    expect(await screen.findByText("Ada's ThinkPad")).toBeTruthy();
    expect(screen.getByText("online")).toBeTruthy();
    expect(screen.getByText("10.0.0.4")).toBeTruthy();
    expect(screen.getByText("nmd_7f3a…")).toBeTruthy();
    expect(devicesApi.listDevices).toHaveBeenCalledWith(false);
  });

  it("revokes a device only after the confirmation is answered", async () => {
    render(<DevicesPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /revoke device ada/i }),
    );

    expect(
      await screen.findByText(
        /that computer stops being able to run your jobs/i,
      ),
    ).toBeTruthy();
    expect(devicesApi.revokeDevice).not.toHaveBeenCalled();

    const buttons = await screen.findAllByRole("button", { name: /revoke/i });
    fireEvent.click(buttons[buttons.length - 1]);

    await waitFor(() => {
      expect(devicesApi.revokeDevice).toHaveBeenCalledWith(online.id);
    });
  });

  it("leaves the device alone when the confirmation is dismissed", async () => {
    render(<DevicesPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /revoke device ada/i }),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Keep" }));

    // antd keeps the dismissed popup mounted behind its exit animation, so what
    // is asserted is the effect of dismissing it rather than its disappearance.
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /revoke device ada/i }),
      ).toBeEnabled();
    });
    expect(devicesApi.revokeDevice).not.toHaveBeenCalled();
    expect(screen.getByText("Ada's ThinkPad")).toBeTruthy();
  });

  it("re-reads the list with include_revoked when the toggle is set", async () => {
    render(<DevicesPage />);

    await screen.findByText("Ada's ThinkPad");
    vi.mocked(devicesApi.listDevices).mockResolvedValue([online, revoked]);

    fireEvent.click(screen.getByRole("checkbox", { name: /show revoked/i }));

    await waitFor(() => {
      expect(devicesApi.listDevices).toHaveBeenCalledWith(true);
    });
    expect(await screen.findByText("Old iMac")).toBeTruthy();
    // A revoked device has nothing left to take away.
    expect(
      screen.queryByRole("button", { name: /revoke device old imac/i }),
    ).toBeNull();
  });

  it("says the feature is off rather than blaming the account, on a 404", async () => {
    vi.mocked(devicesApi.listDevices).mockRejectedValue(
      new ApiError("Not found", 404),
    );

    render(<DevicesPage />);

    expect(
      await screen.findByText("Device pairing is off on this deployment"),
    ).toBeTruthy();
    expect(session.navigateToLogin).not.toHaveBeenCalled();
  });

  it("redirects to login when the session is unauthorized", async () => {
    vi.mocked(api.me).mockRejectedValue(new ApiError("Unauthorized", 401));

    render(<DevicesPage />);

    await waitFor(() => {
      expect(session.navigateToLogin).toHaveBeenCalled();
    });
    expect(screen.queryByText("Devices unavailable")).toBeNull();
  });
});
