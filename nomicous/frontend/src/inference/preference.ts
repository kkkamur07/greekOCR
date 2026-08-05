const STORAGE_KEY = "nomicous_inference_preference";

/**
 * Where OCR / segmentation runs.
 *
 * - `auto`: prefer the local helper when it can serve the model, fall back to
 *   the cloud on any local failure.
 * - `local-only`: cloud inference is disabled. Nothing is ever enqueued on the
 *   server; a missing helper or a failed local run surfaces as an error.
 * - `cloud-only`: the helper is never contacted.
 */
export type InferenceRouting = "auto" | "local-only" | "cloud-only";

export const DEFAULT_INFERENCE_ROUTING: InferenceRouting = "auto";

/**
 * Read a stored value into the current three-state vocabulary.
 * The legacy binary preference stored `"cloud"` / `"local"`; `"cloud"` keeps its
 * meaning as `cloud-only`, everything else becomes the `auto` default.
 */
export function normalizeInferenceRouting(
  raw: string | null | undefined,
): InferenceRouting {
  if (raw === "auto" || raw === "local-only" || raw === "cloud-only") {
    return raw;
  }
  if (raw === "cloud") return "cloud-only";
  return DEFAULT_INFERENCE_ROUTING;
}

export function loadInferenceRouting(): InferenceRouting {
  try {
    return normalizeInferenceRouting(localStorage.getItem(STORAGE_KEY));
  } catch {
    return DEFAULT_INFERENCE_ROUTING;
  }
}

export function saveInferenceRouting(routing: InferenceRouting): void {
  try {
    localStorage.setItem(STORAGE_KEY, routing);
  } catch {
    // A blocked localStorage only costs persistence, not the current session.
  }
}

/** False means: never enqueue a cloud job, whatever happens locally. */
export function cloudInferenceEnabled(routing: InferenceRouting): boolean {
  return routing !== "local-only";
}

/** False means: never contact the local helper. */
export function localInferenceEnabled(routing: InferenceRouting): boolean {
  return routing !== "cloud-only";
}

/** Plain-language labels; researchers read these, not the enum values. */
export const INFERENCE_ROUTING_LABELS: Record<InferenceRouting, string> = {
  auto: "Automatic",
  "local-only": "Local only",
  "cloud-only": "Cloud only",
};

export const INFERENCE_ROUTING_HINTS: Record<InferenceRouting, string> = {
  auto: "Runs on this computer when it can, otherwise in the cloud.",
  "local-only":
    "Runs only on this computer. Nothing is sent to the cloud, so a run fails if the helper is not available.",
  "cloud-only": "Always runs in the cloud. This computer is never used.",
};
