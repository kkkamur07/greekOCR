import { useCallback, useMemo, useRef, useState } from "react";
import type {
  LocalInferenceCallbacks,
  LocalRun,
} from "./localInferenceCallbacks";

/**
 * Owns the abort signals of the page's local runs.
 *
 * A page can only be running one local job at a time, so starting a run cancels
 * whichever run was already in flight. That cancellation is *not* a failure to
 * retry in the cloud - the newer run is going to write the same page - so it
 * gets its own signal, separate from the "use cloud for this run" control.
 */
export function useLocalInferenceRuns(
  isModelCached: (registryModelId: string) => boolean | null,
) {
  /** Set while a run is waiting on weights that are not on this machine yet. */
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(
    null,
  );
  const cloudSwitchRef = useRef<AbortController | null>(null);
  const supersedeRef = useRef<AbortController | null>(null);
  // The helper's own /info document already says which weights are on disk, so
  // the callbacks read the latest snapshot instead of issuing another request.
  const isModelCachedRef = useRef(isModelCached);
  isModelCachedRef.current = isModelCached;

  const localInference = useMemo<LocalInferenceCallbacks>(
    () => ({
      startRun: (registryModelId: string): LocalRun => {
        // Whatever was running for this page is replaced, not raced with.
        supersedeRef.current?.abort();
        const superseded = new AbortController();
        const cloudSwitch = new AbortController();
        supersedeRef.current = superseded;
        cloudSwitchRef.current = cloudSwitch;
        // Only surface the download banner the first time a model is used on
        // this machine. Once the weights are cached, the run proceeds silently.
        const cached = isModelCachedRef.current(registryModelId);
        setDownloadingModelId(cached === false ? registryModelId : null);
        return {
          cloudSwitchSignal: cloudSwitch.signal,
          supersededSignal: superseded.signal,
          end: () => {
            // A superseded run unwinds after its successor has already started:
            // it must not tear down the successor or clear its banner.
            if (supersedeRef.current === superseded) {
              supersedeRef.current = null;
            }
            if (cloudSwitchRef.current === cloudSwitch) {
              cloudSwitchRef.current = null;
              setDownloadingModelId(null);
            }
          },
        };
      },
    }),
    [],
  );

  /** Stop running locally and let this one job fall through to the cloud. */
  const abortRunToCloud = useCallback(() => {
    cloudSwitchRef.current?.abort();
    setDownloadingModelId(null);
  }, []);

  return { localInference, abortRunToCloud, downloadingModelId };
}
