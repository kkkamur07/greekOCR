import type { HostEligibility, InferenceTask } from "./types";
import { fetchHelper } from "./helperClient";

export type HelperCatalogModel = {
  registry_model_id: string;
  task: InferenceTask;
  architecture: string;
  device: string;
  host_eligibility: HostEligibility;
  tags: string[];
};

export async function fetchHelperCatalog(): Promise<HelperCatalogModel[]> {
  const response = await fetchHelper("/inference/v1/catalog");
  if (!response.ok) {
    throw new Error("Inference helper catalog is unavailable.");
  }
  const body = (await response.json()) as { models: HelperCatalogModel[] };
  return body.models;
}

export function isModelLocalEligible(
  catalog: HelperCatalogModel[],
  registryModelId: string,
): boolean {
  const entry = catalog.find(
    (model) => model.registry_model_id === registryModelId,
  );
  if (!entry) return false;
  return entry.host_eligibility === "local" || entry.host_eligibility === "any";
}

export function isModelRemoteOnly(
  catalog: HelperCatalogModel[],
  registryModelId: string,
): boolean {
  const entry = catalog.find(
    (model) => model.registry_model_id === registryModelId,
  );
  return entry?.host_eligibility === "remote";
}
