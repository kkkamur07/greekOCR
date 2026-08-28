import { useEffect, useRef, useState } from "react";

import {
  AGENT_INSTALL_COMMAND,
  AGENT_INSTALL_COMMAND_PIP,
  AGENT_PAIR_COMMAND,
  AGENT_RUN_COMMAND,
} from "../../inference/constants";
import { HOST_PREFERENCE_HINT } from "../../inference/hostPreference";

/**
 * What this computer can currently take, and nothing about where a job went.
 *
 * Nothing here can fail a run: with `local_only` retired, no setting means an
 * absent agent implies "your work does not happen", so absence reads as an
 * ordinary announced state (ADR 0002).
 *
 * "Running" is the platform's answer, not this browser's: it is **capacity**
 * on the account (a device seen recently), the same fact submission uses to
 * fix an **execution target**. A loopback probe could disagree with it.
 */
function agentStatusText(
  preferLocalInference: boolean,
  loading: boolean,
  hasLocalCapacity: boolean,
): string {
  if (!preferLocalInference) return HOST_PREFERENCE_HINT;
  if (loading) return "Checking whether the nomikos agent is running…";
  if (hasLocalCapacity) return "The agent is running on this computer.";
  return "The agent is not running on this computer, so jobs go to the cloud.";
}

type AgentStep = {
  id: string;
  lead: string;
  command: string;
  note?: string;
};

/**
 * Three commands, in the order they are run.
 *
 * Deliberately not four per-OS download buttons: there is one **published
 * package** and a hosted worker installs the same one, so a platform picker
 * here would be describing a distinction that does not exist.
 */
const AGENT_STEPS: AgentStep[] = [
  {
    id: "install",
    lead: "Install the agent",
    command: AGENT_INSTALL_COMMAND,
    note: `Needs uv 0.10 or newer. With pip instead: ${AGENT_INSTALL_COMMAND_PIP}`,
  },
  {
    id: "pair",
    lead: "Link this computer to your account",
    command: AGENT_PAIR_COMMAND,
    note: "It prints a code and a link. Approve it only if the code on the page matches the one in your terminal.",
  },
  {
    id: "run",
    lead: "Start taking work",
    command: AGENT_RUN_COMMAND,
    note: "Leave it running while you work. Closing it sends the next job to the cloud instead.",
  },
];

type PageEditorInferenceBannerProps = {
  /** **Capacity** for this account's own computer, as the platform reports it. */
  hasLocalCapacity: boolean;
  loading: boolean;
  /**
   * The account-level **host preference**, read only. The one control that
   * changes it lives in editor settings; a second copy here would read as a
   * per-run choice, which is exactly what ADR 0002 refuses to have.
   */
  preferLocalInference: boolean;
  onRetry: () => void;
  onUseCloudInstead: () => void;
};

/**
 * Where a viewer's dismissal of the idle hint is remembered.
 *
 * Per browser rather than on the account: it carries no meaning for anyone
 * else, and a round trip to store "I have read one sentence" would be a
 * strange thing to spend a request on.
 */
const HINT_DISMISSED_KEY = "nomikos.inferenceHintDismissed";

function readHintDismissed(): boolean {
  try {
    return window.localStorage.getItem(HINT_DISMISSED_KEY) === "1";
  } catch {
    // A private window, or a browser set to block site data. Not knowing means
    // showing the hint, which is the state a first-time reader gets anyway.
    return false;
  }
}

export function PageEditorInferenceBanner({
  hasLocalCapacity,
  loading,
  preferLocalInference,
  onRetry,
  onUseCloudInstead,
}: PageEditorInferenceBannerProps) {
  const bannerRef = useRef<HTMLDivElement | null>(null);
  const titleId = "pe-agent-install-title";
  const [modalOpen, setModalOpen] = useState(false);
  // Read on mount, not during the first render: the server has no
  // `localStorage`, and reading it inline would hydrate to different markup.
  // `null` means "not read yet" so the banner is withheld for that first frame
  // rather than shown and then yanked, which shifted the whole editor column.
  const [hintDismissed, setHintDismissed] = useState<boolean | null>(null);

  useEffect(() => {
    setHintDismissed(readHintDismissed());
  }, []);

  const shouldPrompt = !loading && !hasLocalCapacity && preferLocalInference;

  /**
   * Only the idle hint can be dismissed. With the preference on, this line is
   * live status - whether the agent is up, and a Retry beside it - and hiding
   * that would leave a researcher wondering where their jobs went.
   */
  const dismissable = !preferLocalInference;

  function handleDismissHint() {
    // The button is about to unmount with the banner around it. Without this,
    // focus falls to <body> and the next Tab restarts from the top of the
    // editor with nothing announced.
    bannerRef.current?.closest("main")?.focus?.();
    setHintDismissed(true);
    try {
      window.localStorage.setItem(HINT_DISMISSED_KEY, "1");
    } catch {
      // Dismissed for this page either way; it just will not be remembered.
    }
  }

  useEffect(() => {
    if (!shouldPrompt) setModalOpen(false);
  }, [shouldPrompt]);

  useEffect(() => {
    if (!modalOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setModalOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [modalOpen]);

  function handleUseCloud() {
    setModalOpen(false);
    onUseCloudInstead();
  }

  function handleNotNow() {
    setModalOpen(false);
  }

  return (
    <>
      {modalOpen ? (
        <div
          className="modal-overlay pe-agent-install-overlay"
          role="presentation"
          onClick={(event) => {
            if (event.target === event.currentTarget) handleNotNow();
          }}
        >
          <div
            className="modal-panel pe-agent-install-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id={titleId}>Run inference on this computer</h2>
            <p className="pe-agent-install-modal__lead">
              Run OCR and segmentation on your own computer&apos;s CPU instead
              of waiting for a hosted worker. Page images are stored in nomikos
              either way, and your browser downloads them from there.
            </p>
            <p className="pe-agent-install-modal__lead">
              It is three commands in a terminal. The same three on macOS,
              Windows and Linux.
            </p>
            <ol className="pe-agent-install-modal__steps">
              {AGENT_STEPS.map((step) => (
                <li key={step.id}>
                  <span className="pe-agent-install-modal__lead-in">
                    {step.lead}
                  </span>
                  <code className="pe-agent-install-modal__command">
                    {step.command}
                  </code>
                  {step.note ? (
                    <span className="pe-agent-install-modal__note">
                      {step.note}
                    </span>
                  ) : null}
                </li>
              ))}
            </ol>
            <p className="pe-agent-install-modal__note">
              This page notices on its own once the agent starts claiming - keep
              it open, or press Retry.
            </p>
            <div className="pe-agent-install-modal__actions">
              <button
                type="button"
                className="btn btn-ghost btn-block"
                onClick={handleUseCloud}
              >
                Use cloud inference instead
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-block"
                onClick={handleNotNow}
              >
                Not now
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {dismissable && hintDismissed ? null : (
        <div
          className="pe-inference-banner"
          role="group"
          aria-label="Where inference runs"
        >
          <span className="pe-inference-banner__status" role="status">
            {agentStatusText(preferLocalInference, loading, hasLocalCapacity)}{" "}
            {preferLocalInference ? (
              <button
                type="button"
                className="pe-inference-banner__action"
                onClick={onRetry}
                disabled={loading}
              >
                {loading ? "Checking…" : "Retry"}
              </button>
            ) : null}{" "}
            {shouldPrompt ? (
              <button
                type="button"
                className="pe-inference-banner__action"
                onClick={() => setModalOpen(true)}
              >
                How to run it here
              </button>
            ) : null}
          </span>
          {dismissable ? (
            <button
              type="button"
              className="pe-inference-banner__dismiss"
              onClick={handleDismissHint}
              aria-label="Dismiss this note"
              title="Dismiss"
            >
              ×
            </button>
          ) : null}
        </div>
      )}
    </>
  );
}
