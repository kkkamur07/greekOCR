import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ApiError } from "../api/errors";
import {
  devicesApi,
  invalidateAfter,
  type DeviceResponse,
  type PairingRequestResponse,
} from "../api/resources";
import { userFacingMessage } from "../api/userFacingError";
import { isUnauthorized } from "../auth/session";
import { AuthFormWrap, AuthLayout } from "../components/layout/AuthLayout";
import { PairingConfirmationCode } from "../components/devices/PairingConfirmationCode";
import {
  clearStashedPairingToken,
  stashedPairingToken,
  takePairingTokenFromUrl,
} from "../components/devices/pairingToken";
import { toast } from "../components/ui/toast";

/**
 * What the researcher is being asked, and what came of it.
 *
 * `invalid` is deliberately one state and not four. The lookup answers unknown,
 * expired, consumed and denied with one indistinguishable 404, so that a token
 * cannot be probed for liveness; repeating that distinction in the UI would
 * hand back exactly what the server withheld.
 */
type ConsentView =
  | { kind: "loading" }
  | { kind: "no-token" }
  | { kind: "invalid" }
  | { kind: "unreachable"; message: string }
  | { kind: "ready"; pairing: PairingRequestResponse }
  | { kind: "approved"; device: DeviceResponse }
  | { kind: "denied"; deviceName: string };

function formatWhen(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-3" style={{ padding: "6px 0" }}>
      <span
        className="text-muted text-sm"
        style={{ minWidth: "9rem", flexShrink: 0 }}
      >
        {label}
      </span>
      <span className="text-sm" style={{ wordBreak: "break-word" }}>
        {value}
      </span>
    </div>
  );
}

export function PairPage() {
  const [view, setView] = useState<ConsentView>({ kind: "loading" });
  const [token, setToken] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<"approve" | "deny" | null>(null);

  /**
   * Read straight through `devicesApi` rather than `useServerQuery`: this is a
   * POST carrying a one-shot secret, and the cache layer's refetch-on-focus
   * would re-send it every time the researcher tabs back from their terminal.
   */
  const lookup = useCallback(async (verificationToken: string) => {
    setView({ kind: "loading" });
    try {
      const pairing = await devicesApi.lookupPairing(verificationToken);
      setView({ kind: "ready", pairing });
    } catch (err) {
      if (isUnauthorized(err)) {
        // The API layer has already started a full-page redirect to login.
        // Anything rendered here would only flash on the way out.
        return;
      }
      if (err instanceof ApiError && err.status === 404) {
        clearStashedPairingToken();
        setView({ kind: "invalid" });
        return;
      }
      setView({
        kind: "unreachable",
        message: userFacingMessage(err, "We could not reach the server."),
      });
    }
  }, []);

  useEffect(() => {
    const verificationToken =
      takePairingTokenFromUrl() ?? stashedPairingToken();
    if (!verificationToken) {
      setView({ kind: "no-token" });
      return;
    }
    setToken(verificationToken);
    void lookup(verificationToken);
  }, [lookup]);

  const decide = async (
    decision: "approve" | "deny",
    pairing: PairingRequestResponse,
  ) => {
    if (!token) return;
    setSubmitting(decision);
    try {
      if (decision === "approve") {
        const device = await devicesApi.approvePairing(
          pairing.pairing_id,
          token,
        );
        clearStashedPairingToken();
        invalidateAfter.deviceListChanged();
        setView({ kind: "approved", device });
      } else {
        await devicesApi.denyPairing(pairing.pairing_id, token);
        clearStashedPairingToken();
        setView({ kind: "denied", deviceName: pairing.device_name });
      }
    } catch (err) {
      if (isUnauthorized(err)) return;
      if (err instanceof ApiError && err.status === 404) {
        clearStashedPairingToken();
        setView({ kind: "invalid" });
        return;
      }
      toast.error(
        userFacingMessage(
          err,
          decision === "approve"
            ? "Could not approve this device."
            : "Could not deny this request.",
        ),
      );
    } finally {
      setSubmitting(null);
    }
  };

  function renderView() {
    switch (view.kind) {
      case "loading":
        return (
          <div role="status" aria-busy="true">
            <h1>Pair a device</h1>
            <p className="auth-sub">Checking this request…</p>
          </div>
        );

      case "no-token":
        return (
          <div>
            <h1>Nothing to approve</h1>
            <p className="auth-sub">
              This link is missing the part that identifies the request.
            </p>
            <div className="notice-banner" role="alert">
              <strong>Open the link your helper printed</strong>
              Copy the whole address, including everything after the{" "}
              <code>#</code>. Starting the pairing again from your terminal
              gives you a fresh link.
            </div>
            <p className="auth-footer-link">
              <Link href="/settings/devices">See your paired devices</Link>
            </p>
          </div>
        );

      case "invalid":
        return (
          <div>
            <h1>This request is no longer valid</h1>
            <p className="auth-sub">
              Pairing requests are short-lived and can only be answered once.
            </p>
            <div className="notice-banner" role="alert">
              <strong>Start a new pairing from your terminal</strong>
              The helper will print a fresh link and a fresh confirmation code.
            </div>
            <p className="auth-footer-link">
              <Link href="/settings/devices">See your paired devices</Link>
            </p>
          </div>
        );

      case "unreachable":
        return (
          <div>
            <h1>Pair a device</h1>
            <p className="auth-sub">We could not load this request.</p>
            <div className="notice-banner" role="alert">
              <strong>Something went wrong</strong>
              {view.message}
            </div>
            <button
              type="button"
              className="btn btn-primary btn-block mt-4"
              onClick={() => {
                if (token) void lookup(token);
              }}
            >
              Try again
            </button>
          </div>
        );

      case "approved":
        return (
          <div>
            <h1>{view.device.name} is paired</h1>
            <p className="auth-sub">
              Go back to your terminal - the helper picks this up on its own
              within a few seconds.
            </p>
            <div className="notice-inline">
              <strong>{view.device.platform}</strong>
              <span className="spacer" />
              <span className="text-muted">
                helper {view.device.helper_version}
              </span>
            </div>
            <p className="auth-footer-link">
              Changed your mind?{" "}
              <Link href="/settings/devices">Revoke it from your devices</Link>
            </p>
          </div>
        );

      case "denied":
        return (
          <div>
            <h1>Request denied</h1>
            <p className="auth-sub">
              {view.deviceName} was not paired, and no token was issued for it.
            </p>
            <div className="notice-banner">
              <strong>If you did not start this request</strong>
              Someone else asked to pair a computer with your account. Nothing
              was granted, and there is nothing further you need to do.
            </div>
            <p className="auth-footer-link">
              <Link href="/settings/devices">See your paired devices</Link>
            </p>
          </div>
        );

      case "ready": {
        const { pairing } = view;
        return (
          <div>
            <h1>Approve this computer?</h1>
            <p className="auth-sub">
              It will be able to run jobs on your account until you revoke it.
            </p>

            <PairingConfirmationCode code={pairing.confirmation_code} />

            <DetailRow label="Device name" value={pairing.device_name} />
            <DetailRow label="Platform" value={pairing.platform} />
            <DetailRow label="Helper version" value={pairing.helper_version} />
            <DetailRow
              label="Requested"
              value={formatWhen(pairing.requested_at)}
            />
            <DetailRow label="Expires" value={formatWhen(pairing.expires_at)} />

            <div className="flex gap-3 mt-4">
              <button
                type="button"
                className="btn btn-primary btn-block"
                disabled={submitting !== null}
                onClick={() => void decide("approve", pairing)}
              >
                {submitting === "approve" ? "Approving…" : "Approve"}
              </button>
              <button
                type="button"
                className="btn btn-outline btn-block btn--danger-ghost"
                disabled={submitting !== null}
                onClick={() => void decide("deny", pairing)}
              >
                {submitting === "deny" ? "Denying…" : "Deny"}
              </button>
            </div>
          </div>
        );
      }
    }
  }

  return (
    <AuthLayout
      headline="Pair a computer with your account"
      tagline="A paired computer can run segmentation and transcription jobs for you. Approve one only when you started the request yourself, and only when the code on screen matches the one your terminal printed."
    >
      <AuthFormWrap>{renderView()}</AuthFormWrap>
    </AuthLayout>
  );
}
