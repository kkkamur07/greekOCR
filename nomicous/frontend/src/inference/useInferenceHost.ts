import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchHelperInfo,
  modelCacheState,
  sameHelperModels,
  shouldRunOnLocalHelper,
  NO_HELPER_MODELS,
  type HelperModelInfo,
} from "./helperInfo";
import {
  cloudInferenceEnabled,
  loadInferenceRouting,
  saveInferenceRouting,
  type InferenceRouting,
} from "./preference";

export function useInferenceHost() {
  const [helperAvailable, setHelperAvailable] = useState(false);
  const [helperVersion, setHelperVersion] = useState<string | null>(null);
  const [models, setModels] = useState<HelperModelInfo[]>(NO_HELPER_MODELS);
  const [routing, setRouting] =
    useState<InferenceRouting>(loadInferenceRouting);
  const [probing, setProbing] = useState(true);
  const probingRef = useRef(false);
  const routingRef = useRef(routing);
  routingRef.current = routing;

  const refresh = useCallback(async (options?: { quiet?: boolean }) => {
    if (probingRef.current) return;
    probingRef.current = true;
    if (!options?.quiet) {
      setProbing(true);
    }
    try {
      // "Cloud only" must not touch the loopback port at all.
      const info =
        routingRef.current === "cloud-only" ? null : await fetchHelperInfo();
      setHelperAvailable(info !== null);
      setHelperVersion(info?.version ?? null);
      const nextModels = info?.models ?? NO_HELPER_MODELS;
      // Replace state only when the content actually changed: a fresh array on
      // every probe would re-render the whole editor for nothing.
      setModels((current) =>
        sameHelperModels(current, nextModels) ? current : nextModels,
      );
    } finally {
      probingRef.current = false;
      if (!options?.quiet) {
        setProbing(false);
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, routing]);

  useEffect(() => {
    // Re-check when the user returns to the tab - they may have just started the
    // helper. Everything in between is covered by the explicit Retry control,
    // not by a background timer.
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
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refresh]);

  const setInferenceRouting = useCallback((next: InferenceRouting) => {
    setRouting(next);
    saveInferenceRouting(next);
  }, []);

  function shouldUseLocalPath(registryModelId: string): boolean {
    return shouldRunOnLocalHelper(models, registryModelId, {
      helperAvailable,
      routing,
    });
  }

  /** `true` / `false` when the helper listed the model, `null` when unknown. */
  function isModelCached(registryModelId: string): boolean | null {
    return modelCacheState(models, registryModelId);
  }

  return {
    helperAvailable,
    helperVersion,
    models,
    routing,
    cloudEnabled: cloudInferenceEnabled(routing),
    probing,
    refresh,
    setInferenceRouting,
    shouldUseLocalPath,
    isModelCached,
  };
}
