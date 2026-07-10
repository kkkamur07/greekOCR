import { useCallback, useEffect, useState } from "react";
import {
  fetchHelperCatalog,
  isModelLocalEligible,
  type HelperCatalogModel,
} from "./catalog";
import {
  preferCloudInference,
  saveInferencePreference,
  type InferencePreference,
} from "./preference";
import { probeHelperHealth } from "./probe";

export function useInferenceHost() {
  const [helperAvailable, setHelperAvailable] = useState(false);
  const [catalog, setCatalog] = useState<HelperCatalogModel[]>([]);
  const [preference, setPreference] = useState<InferencePreference>(() =>
    preferCloudInference() ? "cloud" : "local",
  );
  const [probing, setProbing] = useState(true);

  const refresh = useCallback(async () => {
    setProbing(true);
    try {
      const healthy = await probeHelperHealth();
      setHelperAvailable(healthy);
      if (healthy) {
        setCatalog(await fetchHelperCatalog());
      } else {
        setCatalog([]);
      }
    } catch {
      setHelperAvailable(false);
      setCatalog([]);
    } finally {
      setProbing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  function setInferencePreference(next: InferencePreference) {
    setPreference(next);
    saveInferencePreference(next);
  }

  function shouldUseLocalPath(registryModelId: string): boolean {
    if (preference === "cloud" || !helperAvailable) return false;
    return isModelLocalEligible(catalog, registryModelId);
  }

  return {
    helperAvailable,
    catalog,
    preference,
    preferCloud: preference === "cloud",
    probing,
    refresh,
    setInferencePreference,
    shouldUseLocalPath,
  };
}
