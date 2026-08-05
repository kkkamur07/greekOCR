import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchHelperInfo,
  modelCacheState,
  sameHelperModels,
  shouldRunOnLocalHelper,
  NO_HELPER_MODELS,
  type HelperModelInfo,
} from "./helperInfo";
import { useHostPreference } from "./hostPreference";

/**
 * What this browser can see of the **inference host**s, joined to the account's
 * **host preference**.
 *
 * The loopback probe below is the pre-ADR-0002 transport and is issue 060's to
 * delete. What changed here is only its input: it is driven by the account
 * setting rather than by a three-way per-browser routing mode.
 */
export function useInferenceHost() {
  const [helperAvailable, setHelperAvailable] = useState(false);
  const [helperVersion, setHelperVersion] = useState<string | null>(null);
  const [models, setModels] = useState<HelperModelInfo[]>(NO_HELPER_MODELS);
  const hostPreference = useHostPreference();
  const [probing, setProbing] = useState(true);
  const probingRef = useRef(false);
  const preferLocalRef = useRef(hostPreference.preferLocalInference);
  preferLocalRef.current = hostPreference.preferLocalInference;

  const refresh = useCallback(async (options?: { quiet?: boolean }) => {
    if (probingRef.current) return;
    probingRef.current = true;
    if (!options?.quiet) {
      setProbing(true);
    }
    try {
      // An account that has not asked for its own computer must not touch the
      // loopback port at all.
      const info = preferLocalRef.current ? await fetchHelperInfo() : null;
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
  }, [refresh, hostPreference.preferLocalInference]);

  useEffect(() => {
    // Re-check when the user returns to the tab - they may have just started the
    // agent. Everything in between is covered by the explicit Retry control,
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

  function shouldUseLocalPath(registryModelId: string): boolean {
    return shouldRunOnLocalHelper(models, registryModelId, {
      helperAvailable,
      preferLocalInference: hostPreference.preferLocalInference,
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
    preferLocalInference: hostPreference.preferLocalInference,
    availableTargets: hostPreference.preference?.available_targets ?? [],
    preferenceLoading: hostPreference.loading,
    preferenceSaving: hostPreference.saving,
    preferenceError: hostPreference.error,
    setPreferLocalInference: hostPreference.setPreferLocalInference,
    probing,
    refresh,
    shouldUseLocalPath,
    isModelCached,
  };
}
