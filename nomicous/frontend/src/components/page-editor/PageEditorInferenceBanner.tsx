import { useEffect, useId, useState } from "react";

import {
  INFERENCE_HELPER_LINUX_TARBALL_URL,
  INFERENCE_HELPER_MACOS_DMG_URL,
  INFERENCE_HELPER_RELEASES_URL,
  INFERENCE_HELPER_WINDOWS_ZIP_URL,
} from "../../inference/constants";

type HelperPlatform = "macos" | "windows" | "linux";

type HelperDownload = {
  platform: HelperPlatform;
  label: string;
  url: string;
};

const HELPER_DOWNLOADS: HelperDownload[] = [
  {
    platform: "macos",
    label: "Download for macOS",
    url: INFERENCE_HELPER_MACOS_DMG_URL,
  },
  {
    platform: "windows",
    label: "Download for Windows",
    url: INFERENCE_HELPER_WINDOWS_ZIP_URL,
  },
  {
    platform: "linux",
    label: "Download for Linux",
    url: INFERENCE_HELPER_LINUX_TARBALL_URL,
  },
];

function detectPlatform(): HelperPlatform | null {
  if (typeof navigator === "undefined") return null;
  const hint =
    `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`.toLowerCase();
  if (hint.includes("mac")) return "macos";
  if (hint.includes("win")) return "windows";
  if (hint.includes("linux") || hint.includes("x11")) return "linux";
  return null;
}

type PageEditorInferenceBannerProps = {
  helperAvailable: boolean;
  probing: boolean;
  preferCloud: boolean;
  onUseCloudInstead: () => void;
};

export function PageEditorInferenceBanner({
  helperAvailable,
  probing,
  preferCloud,
  onUseCloudInstead,
}: PageEditorInferenceBannerProps) {
  const titleId = useId();
  const [modalOpen, setModalOpen] = useState(false);

  const detected = detectPlatform();
  const downloads = detected
    ? [...HELPER_DOWNLOADS].sort((a, b) => {
        if (a.platform === detected) return -1;
        if (b.platform === detected) return 1;
        return 0;
      })
    : HELPER_DOWNLOADS;

  const shouldPrompt = !probing && !helperAvailable && !preferCloud;

  useEffect(() => {
    if (!shouldPrompt) {
      setModalOpen(false);
    }
  }, [shouldPrompt]);

  useEffect(() => {
    if (!modalOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setModalOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [modalOpen]);

  if (!shouldPrompt) {
    return null;
  }

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
          className="modal-overlay pe-helper-install-overlay"
          role="presentation"
          onClick={(event) => {
            if (event.target === event.currentTarget) handleNotNow();
          }}
        >
          <div
            className="modal-panel pe-helper-install-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id={titleId}>Install Inference Helper</h2>
            <p className="pe-helper-install-modal__lead">
              Run OCR and segmentation on your computer&apos;s CPU — faster and
              private. The helper runs in the background after you install it.
            </p>
            <ol className="pe-helper-install-modal__steps">
              <li>
                Download the installer for your operating system from GitHub
                releases.
              </li>
              <li>
                Install <strong>Nomicous Inference Helper</strong> and launch it
                once.
              </li>
              <li>Refresh this page.</li>
            </ol>
            <div className="pe-helper-install-modal__actions">
              {downloads.map((download, index) => (
                <a
                  key={download.platform}
                  href={download.url}
                  target="_blank"
                  rel="noreferrer"
                  className={
                    index === 0
                      ? "btn btn-primary btn-block"
                      : "btn btn-ghost btn-block"
                  }
                >
                  {download.label}
                </a>
              ))}
              <a
                href={INFERENCE_HELPER_RELEASES_URL}
                target="_blank"
                rel="noreferrer"
                className="btn btn-ghost btn-block"
              >
                View release notes
              </a>
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
      ) : (
        <div className="pe-inference-banner" role="status">
          <span>
            Local inference is faster with the Nomicous Inference Helper
            installed on this computer.
          </span>
          <button
            type="button"
            className="pe-inference-banner__action"
            onClick={() => setModalOpen(true)}
          >
            Install helper
          </button>
        </div>
      )}
    </>
  );
}
