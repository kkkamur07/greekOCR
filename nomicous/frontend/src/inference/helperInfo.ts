import {
  HELPER_INFO_PATH,
  HELPER_PROBE_TIMEOUT_MS,
  HELPER_SERVICE_NAME,
} from "./constants";
import { fetchHelper } from "./helperClient";
import type { InferenceRouting } from "./preference";
import type { HostEligibility, InferenceTask } from "./types";

export type HelperModelInfo = {
  registry_model_id: string;
  task: InferenceTask;
  host_eligibility: HostEligibility;
  tags: string[];
  cached: boolean;
};

export type HelperInfo = {
  service: typeof HELPER_SERVICE_NAME;
  version: string;
  models: HelperModelInfo[];
};

/** Stable empty list so "no helper" refreshes do not allocate a new array. */
export const NO_HELPER_MODELS: HelperModelInfo[] = [];

function parseModel(entry: unknown): HelperModelInfo | null {
  if (typeof entry !== "object" || entry === null) return null;
  const record = entry as Record<string, unknown>;
  if (typeof record.registry_model_id !== "string") return null;
  if (typeof record.task !== "string") return null;
  if (typeof record.host_eligibility !== "string") return null;
  return {
    registry_model_id: record.registry_model_id,
    task: record.task as InferenceTask,
    host_eligibility: record.host_eligibility as HostEligibility,
    tags: Array.isArray(record.tags)
      ? record.tags.filter((tag): tag is string => typeof tag === "string")
      : [],
    cached: record.cached === true,
  };
}

/**
 * Accept a response only when it identifies itself as the Nomicous helper.
 *
 * Any other body - a foreign dev server answering on the same port, an HTML
 * error page, a JSON document without `service` - is treated as "no helper",
 * because the next step after this check is POSTing a manuscript image.
 */
export function parseHelperInfo(body: unknown): HelperInfo | null {
  if (typeof body !== "object" || body === null) return null;
  const record = body as Record<string, unknown>;
  if (record.service !== HELPER_SERVICE_NAME) return null;
  const models = Array.isArray(record.models) ? record.models : [];
  return {
    service: HELPER_SERVICE_NAME,
    version: typeof record.version === "string" ? record.version : "",
    models: models
      .map(parseModel)
      .filter((model): model is HelperModelInfo => model !== null),
  };
}

/**
 * One document describes the helper: is it there, which build, which models it
 * can serve and whether their weights are already on disk.
 *
 * Returns `null` whenever the helper cannot be verified. Callers must treat
 * `null` as "helper absent" and never fall back to a bare reachability check.
 */
export async function fetchHelperInfo(): Promise<HelperInfo | null> {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    HELPER_PROBE_TIMEOUT_MS,
  );
  try {
    const response = await fetchHelper(HELPER_INFO_PATH, {
      method: "GET",
      signal: controller.signal,
    });
    if (!response.ok) return null;
    return parseHelperInfo(await response.json());
  } catch {
    return null;
  } finally {
    window.clearTimeout(timeout);
  }
}

function findModel(
  models: HelperModelInfo[],
  registryModelId: string,
): HelperModelInfo | undefined {
  return models.find((model) => model.registry_model_id === registryModelId);
}

export function isModelLocalEligible(
  models: HelperModelInfo[],
  registryModelId: string,
): boolean {
  const entry = findModel(models, registryModelId);
  if (!entry) return false;
  return entry.host_eligibility === "local" || entry.host_eligibility === "any";
}

export function isModelRemoteOnly(
  models: HelperModelInfo[],
  registryModelId: string,
): boolean {
  return findModel(models, registryModelId)?.host_eligibility === "remote";
}

/**
 * Whether the helper already holds this model's weights.
 * `null` means "unknown" (helper absent or model not listed), which callers use
 * to avoid flashing a false "Downloading…" banner.
 */
export function modelCacheState(
  models: HelperModelInfo[],
  registryModelId: string,
): boolean | null {
  const entry = findModel(models, registryModelId);
  return entry ? entry.cached : null;
}

/**
 * Decide whether a run should hit the local helper.
 *
 * Local-only catalog entries (`host_eligibility: "local"`, e.g. blla-segment)
 * always use the helper when it is up - a "cloud only" routing choice must not
 * enqueue a cloud job that can never claim them, so `cloud-only` instead means
 * "never call the helper at all".
 */
export function shouldRunOnLocalHelper(
  models: HelperModelInfo[],
  registryModelId: string,
  options: { helperAvailable: boolean; routing: InferenceRouting },
): boolean {
  if (options.routing === "cloud-only") return false;
  if (!options.helperAvailable) return false;
  const entry = findModel(models, registryModelId);
  if (!entry) return false;
  return entry.host_eligibility !== "remote";
}

/** Content comparison so a poll that changed nothing does not re-render. */
export function sameHelperModels(
  left: HelperModelInfo[],
  right: HelperModelInfo[],
): boolean {
  if (left === right) return true;
  if (left.length !== right.length) return false;
  return left.every((model, index) => {
    const other = right[index];
    return (
      model.registry_model_id === other.registry_model_id &&
      model.task === other.task &&
      model.host_eligibility === other.host_eligibility &&
      model.cached === other.cached &&
      model.tags.length === other.tags.length &&
      model.tags.every((tag, tagIndex) => tag === other.tags[tagIndex])
    );
  });
}
