import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  subscribeProjectDefaultWritten,
  whenProjectDefaultSettled,
  type InferenceTask,
  type ModelBindingResponse,
} from "../../api/client";
import { ApiError } from "../../api/errors";

export type ProjectBinding = Pick<
  ModelBindingResponse,
  "id" | "task" | "model_id"
>;

export type BindingStore = {
  list: (projectId: string) => Promise<ProjectBinding[]>;
  create: (
    projectId: string,
    task: InferenceTask,
    modelId: string,
  ) => Promise<ProjectBinding>;
  update: (
    projectId: string,
    bindingId: string,
    modelId: string,
  ) => Promise<ProjectBinding>;
};

export function apiBindingStore(): BindingStore {
  return {
    list: (projectId) => api.listProjectModelBindings(projectId),
    create: (projectId, task, modelId) =>
      api.createProjectModelBinding(projectId, { task, model_id: modelId }),
    update: (projectId, bindingId, modelId) =>
      api.updateProjectModelBinding(projectId, bindingId, {
        model_id: modelId,
      }),
  };
}

/**
 * Explicit "Set as project default": POST when the project has no binding for
 * the task, PATCH the binding when one exists. A 409 from the create means a
 * collaborator saved one in the meantime, so refetch and PATCH that row
 * instead of failing.
 */
export async function saveProjectDefault(
  store: BindingStore,
  projectId: string,
  task: InferenceTask,
  modelId: string,
): Promise<ProjectBinding> {
  const bindings = await store.list(projectId);
  const existing = bindings.find((binding) => binding.task === task);
  if (existing) {
    if (existing.model_id === modelId) return existing;
    try {
      return await store.update(projectId, existing.id, modelId);
    } catch (err) {
      // Deleted between the list and the update; fall through and create.
      if (!(err instanceof ApiError) || err.status !== 404) throw err;
    }
  }
  try {
    return await store.create(projectId, task, modelId);
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 409) throw err;
    const fresh = await store.list(projectId);
    const raced = fresh.find((binding) => binding.task === task);
    if (!raced) throw err;
    return await store.update(projectId, raced.id, modelId);
  }
}

/**
 * The project bindings for the pickers' "Set as project default" controls.
 * A failed load reads as "no defaults", never as an error banner: the pickers
 * keep their binding-or-catalog fallback and the control simply hides.
 */
export function useProjectModelDefaults(
  projectId: string | undefined,
  store?: BindingStore,
) {
  const resolvedStore = useMemo(() => store ?? apiBindingStore(), [store]);
  const [bindings, setBindings] = useState<ProjectBinding[]>([]);
  const [saving, setSaving] = useState<InferenceTask | null>(null);
  const [error, setError] = useState<string | null>(null);
  /**
   * Which load is newest. A save that lands while the initial list is still
   * in flight must win over it: without this the late list would overwrite
   * the just-saved default with the server rows it left behind.
   */
  const generationRef = useRef(0);

  useEffect(() => {
    if (!projectId) {
      setBindings([]);
      return;
    }
    let cancelled = false;
    const generation = ++generationRef.current;
    const isCurrent = () => !cancelled && generationRef.current === generation;
    // Wrapped so a store that throws (or a test double that answers with
    // nothing) reads as "no defaults" rather than crashing the picker.
    void Promise.resolve()
      .then(() => resolvedStore.list(projectId))
      .then(
        (rows) => {
          if (isCurrent()) setBindings(Array.isArray(rows) ? rows : []);
        },
        () => {
          if (isCurrent()) setBindings([]);
        },
      );
    return () => {
      cancelled = true;
    };
  }, [projectId, resolvedStore]);

  const defaultModelId = useCallback(
    (task: InferenceTask): string | null =>
      bindings.find((binding) => binding.task === task)?.model_id ?? null,
    [bindings],
  );

  /**
   * Re-read the project bindings, after waiting out the automatic write in
   * flight if any. A late initial list never wins over newer state: every
   * read carries the same generation guard as the mount load.
   */
  const refresh = useCallback(async (): Promise<void> => {
    if (!projectId) return;
    const generation = ++generationRef.current;
    const isCurrent = () => generationRef.current === generation;
    await whenProjectDefaultSettled();
    try {
      const rows = await resolvedStore.list(projectId);
      if (isCurrent()) setBindings(Array.isArray(rows) ? rows : []);
    } catch {
      if (isCurrent()) setBindings([]);
    }
  }, [projectId, resolvedStore]);

  // The automatic write lives in the API client, outside this state, so the
  // client announces each stored binding and this re-reads. No polling, no
  // timers: the notification fires once the write completes.
  useEffect(() => {
    if (!projectId) return;
    return subscribeProjectDefaultWritten((info) => {
      if (info.projectId !== projectId) return;
      void refresh();
    });
  }, [projectId, refresh]);

  const saveDefault = useCallback(
    async (
      task: InferenceTask,
      modelId: string,
    ): Promise<ProjectBinding | null> => {
      if (!projectId) return null;
      setSaving(task);
      setError(null);
      try {
        const saved = await saveProjectDefault(
          resolvedStore,
          projectId,
          task,
          modelId,
        );
        // A save outranks an initial list still in flight; see generationRef.
        generationRef.current += 1;
        setBindings((current) => [
          ...current.filter((binding) => binding.task !== task),
          saved,
        ]);
        return saved;
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Could not save the project default.",
        );
        return null;
      } finally {
        setSaving(null);
      }
    },
    [projectId, resolvedStore],
  );

  return { bindings, saving, error, defaultModelId, saveDefault, refresh };
}
