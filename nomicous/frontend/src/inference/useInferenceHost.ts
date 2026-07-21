import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchHelperCatalog,
  shouldRunOnLocalHelper,
  type HelperCatalogModel,
} from "./catalog";
import { HELPER_PROBE_INTERVAL_MS } from "./constants";
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
  const probingRef = useRef(false);

  const refresh = useCallback(async (options?: { quiet?: boolean }) => {
    if (probingRef.current) return;
    probingRef.current = true;
    if (!options?.quiet) {
      setProbing(true);
    }
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
      probingRef.current = false;
      if (!options?.quiet) {
        setProbing(false);
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    function onFocus() {
      void refresh({ quiet: true });
    }
    function onVisibility() {
      if (document.visibilityState === "visible") {
        void refresh({ quiet: true });
      }
    }
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    const interval = window.setInterval(() => {
      void refresh({ quiet: true });
    }, HELPER_PROBE_INTERVAL_MS);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
      window.clearInterval(interval);
    };
  }, [refresh]);

  function setInferencePreference(next: InferencePreference) {
    setPreference(next);
    saveInferencePreference(next);
  }

  function shouldUseLocalPath(registryModelId: string): boolean {
    return shouldRunOnLocalHelper(catalog, registryModelId, {
      helperAvailable,
      preferCloud: preference === "cloud",
    });
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
