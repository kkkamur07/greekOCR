import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/errors";
import { devicesApi } from "../api/resources";
import { PairPage } from "./PairPage";

vi.mock("../api/resources", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/resources")>();
  return {
    ...actual,
    devicesApi: {
      ...actual.devicesApi,
      lookupPairing: vi.fn(),
      approvePairing: vi.fn(),
      denyPairing: vi.fn(),
    },
  };
});

const VERIFICATION_TOKEN = "v1.a-long-opaque-consent-token";

const pairing = {
  pairing_id: "9d5a9d54-0000-4000-8000-000000000001",
  device_name: "Ada's ThinkPad",
  platform: "linux-x86_64",
  helper_version: "0.4.1",
  confirmation_code: "K7QF-2M9X",
  requested_at: "2026-08-06T09:00:00Z",
  expires_at: "2026-08-06T09:10:00Z",
};

const device = {
  id: "9d5a9d54-0000-4000-8000-000000000002",
  name: "Ada's ThinkPad",
  platform: "linux-x86_64",
  helper_version: "0.4.1",
  status: "pairing" as const,
  token_prefix: "nmd_7f3a",
  paired_at: "2026-08-06T09:01:00Z",
};

function openConsentLink(fragment: string = VERIFICATION_TOKEN) {
  window.history.replaceState({}, "", `/pair#${fragment}`);
}

describe("PairPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    vi.mocked(devicesApi.lookupPairing).mockResolvedValue(pairing);
    vi.mocked(devicesApi.approvePairing).mockResolvedValue(device);
    vi.mocked(devicesApi.denyPairing).mockResolvedValue(undefined);
  });

  it("shows the confirmation code and the device asking to be paired", async () => {
    openConsentLink();

    render(<PairPage />);

    expect(await screen.findByText("K7QF-2M9X")).toBeTruthy();
    expect(screen.getByText("Ada's ThinkPad")).toBeTruthy();
    expect(screen.getByText("linux-x86_64")).toBeTruthy();
    expect(screen.getByText("0.4.1")).toBeTruthy();
    expect(devicesApi.lookupPairing).toHaveBeenCalledWith(VERIFICATION_TOKEN);
  });

  it("takes the token out of the address bar before anything else can read it", async () => {
    openConsentLink();

    render(<PairPage />);

    await screen.findByText("K7QF-2M9X");
    // A fragment never reaches the server on its own; what would leak it is the
    // login redirect folding `location.hash` into a `callbackUrl` query.
    expect(window.location.hash).toBe("");
    expect(window.location.pathname).toBe("/pair");
  });

  it("approves the pairing and sends the researcher back to their terminal", async () => {
    openConsentLink();

    render(<PairPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(devicesApi.approvePairing).toHaveBeenCalledWith(
        pairing.pairing_id,
        VERIFICATION_TOKEN,
      );
    });
    expect(
      await screen.findByRole("heading", { name: "Ada's ThinkPad is paired" }),
    ).toBeTruthy();
    expect(screen.getByText(/go back to your terminal/i)).toBeTruthy();
    // The consent token is single-use; nothing should be left to replay.
    expect(window.sessionStorage.length).toBe(0);
  });

  it("denies the pairing without issuing a token", async () => {
    openConsentLink();

    render(<PairPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Deny" }));

    await waitFor(() => {
      expect(devicesApi.denyPairing).toHaveBeenCalledWith(
        pairing.pairing_id,
        VERIFICATION_TOKEN,
      );
    });
    expect(
      await screen.findByRole("heading", { name: "Request denied" }),
    ).toBeTruthy();
    expect(devicesApi.approvePairing).not.toHaveBeenCalled();
  });

  it("says one thing for every 404, because the server tells it no more", async () => {
    vi.mocked(devicesApi.lookupPairing).mockRejectedValue(
      new ApiError("Not found", 404),
    );
    openConsentLink();

    render(<PairPage />);

    expect(
      await screen.findByRole("heading", {
        name: "This request is no longer valid",
      }),
    ).toBeTruthy();
    expect(
      screen.getByText(/start a new pairing from your terminal/i),
    ).toBeTruthy();
    // Nothing on screen may hint at which of unknown/expired/consumed/denied
    // this was - that distinction is exactly what the 404 withholds.
    expect(screen.queryByText(/expired/i)).toBeNull();
    expect(screen.queryByText(/already used/i)).toBeNull();
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("asks for the whole link when the fragment is missing", async () => {
    window.history.replaceState({}, "", "/pair");

    render(<PairPage />);

    expect(
      await screen.findByRole("heading", { name: "Nothing to approve" }),
    ).toBeTruthy();
    expect(devicesApi.lookupPairing).not.toHaveBeenCalled();
  });

  it("offers a retry when the server cannot be reached, and keeps the token", async () => {
    vi.mocked(devicesApi.lookupPairing).mockRejectedValueOnce(
      new TypeError("Failed to fetch"),
    );
    openConsentLink();

    render(<PairPage />);

    expect(await screen.findByText("Something went wrong")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByText("K7QF-2M9X")).toBeTruthy();
    expect(devicesApi.lookupPairing).toHaveBeenLastCalledWith(
      VERIFICATION_TOKEN,
    );
  });

  it("recovers the token from the tab after a login round trip", async () => {
    // What the route wrapper does while the session is still restoring: the
    // fragment is parked, and the address bar no longer has it.
    openConsentLink();
    const { takePairingTokenFromUrl } =
      await import("../components/devices/pairingToken");
    takePairingTokenFromUrl();
    expect(window.location.hash).toBe("");

    render(<PairPage />);

    expect(await screen.findByText("K7QF-2M9X")).toBeTruthy();
    expect(devicesApi.lookupPairing).toHaveBeenCalledWith(VERIFICATION_TOKEN);
  });

  it("keeps the pairing on screen when a decision fails outright", async () => {
    vi.mocked(devicesApi.approvePairing).mockRejectedValue(
      new ApiError("Something went wrong on the server.", 500),
    );
    openConsentLink();

    render(<PairPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(devicesApi.approvePairing).toHaveBeenCalled();
    });
    // Still answerable: a failed approve must not strand the request.
    expect(screen.getByRole("button", { name: "Approve" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Deny" })).toBeTruthy();
  });
});
