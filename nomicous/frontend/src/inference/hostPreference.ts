/**
 * The account-level **host preference**: "use my computer when it is
 * available".
 *
 * It is the only researcher input to **execution target** selection, and it is
 * chosen once for the account rather than per job - a researcher cannot know
 * which host is faster for a given page, so asking at every action is a
 * decision without a basis, and it is exactly what regrows the three-mode
 * complexity ADR 0002 deletes.
 *
 * The setting lives on the server, not in `localStorage`. The platform is what
 * decides an execution target at submission, so a preference the platform
 * cannot read would be a preference in name only.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, type ExecutionPreferenceResponse } from "../api/client";
import { userFacingMessage } from "../api/userFacingError";

export const HOST_PREFERENCE_LABEL = "Use my computer when it is available";

export const HOST_PREFERENCE_HINT =
  "Jobs run on this computer while the nomicous agent is running, and in the cloud otherwise. Each job says which one it used.";

export type HostPreference = {
  /** `null` until the account setting has been read. */
  preference: ExecutionPreferenceResponse | null;
  preferLocalInference: boolean;
  loading: boolean;
  saving: boolean;
  error: string | null;
  setPreferLocalInference: (preferLocal: boolean) => Promise<void>;
};

export function useHostPreference(): HostPreference {
  const [preference, setPreference] =
    useState<ExecutionPreferenceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void api
      .getExecutionPreference({ signal: controller.signal })
      .then((value) => {
        if (controller.signal.aborted || !mountedRef.current) return;
        setPreference(value);
        setError(null);
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted || !mountedRef.current) return;
        setError(
          userFacingMessage(failure, "Could not read your host setting."),
        );
      })
      .finally(() => {
        if (controller.signal.aborted || !mountedRef.current) return;
        setLoading(false);
      });
    return () => controller.abort();
  }, []);

  /**
   * The response is the new state. Writing what the server returned rather than
   * what was sent is what makes the round trip visible: `available_targets` and
   * `preferred_execution_target` are recomputed from live **capacity**, and a
   * client that echoed its own input would show a preference the platform has
   * not agreed to.
   */
  const setPreferLocalInference = useCallback(async (preferLocal: boolean) => {
    setSaving(true);
    try {
      const saved = await api.setExecutionPreference(preferLocal);
      if (!mountedRef.current) return;
      setPreference(saved);
      setError(null);
    } catch (failure) {
      if (!mountedRef.current) return;
      setError(userFacingMessage(failure, "Could not save your host setting."));
    } finally {
      if (mountedRef.current) setSaving(false);
    }
  }, []);

  return {
    preference,
    // Until the account is read, assume the cloud: it is the host that needs
    // nothing installed, so a wrong guess costs a probe rather than a failure.
    preferLocalInference: preference?.prefer_local_inference ?? false,
    loading,
    saving,
    error,
    setPreferLocalInference,
  };
}
