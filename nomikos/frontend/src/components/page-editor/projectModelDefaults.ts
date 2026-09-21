import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  type InferenceTask,
  type ModelBindingResponse,
} from "../../api/client";
import { ApiError } from "../../api/errors";

/**
 * What a save or a clear answers with. The message travels back to the
 * caller rather than only into `error` state, so the toast at the call site
 * says what the API said instead of reading a value React has not set yet.
 */
export type ProjectDefaultResult =
  { ok: true; binding: ProjectBinding | null } | { ok: false; message: string };

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
  remove: (projectId: string, bindingId: string) => Promise<void>;
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
    remove: (projectId, bindingId) =>
      api.deleteProjectModelBinding(projectId, bindingId),
  };
}

/**
 * Saving a project default: POST when the project has no binding for the
 * task, PATCH the binding when one exists. A 409 from the create means a
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
 * Clearing a project default: delete the binding for the task if there is
 * one. Nothing to delete is success, not an error: a collaborator who
 * cleared it first left the project in exactly the wanted state.
 */
export async function clearProjectDefault(
  store: BindingStore,
  projectId: string,
  task: InferenceTask,
): Promise<void> {
  const bindings = await store.list(projectId);
  const existing = bindings.find((binding) => binding.task === task);
  if (!existing) return;
  try {
    await store.remove(projectId, existing.id);
  } catch (err) {
    // Already gone; the caller wanted no binding and there is none.
    if (!(err instanceof ApiError) || err.status !== 404) throw err;
  }
}

/**
 * The project bindings behind the project page's default models card and
 * behind every picker's preselection.
 * A failed load reads as "no defaults", never as an error banner: the pickers
 * keep their binding-or-catalog fallback.
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
   * Re-read the project bindings. A late initial list never wins over newer
   * state: every read carries the same generation guard as the mount load.
   */
  const refresh = useCallback(async (): Promise<void> => {
    if (!projectId) return;
    const generation = ++generationRef.current;
    const isCurrent = () => generationRef.current === generation;
    try {
      const rows = await resolvedStore.list(projectId);
      if (isCurrent()) setBindings(Array.isArray(rows) ? rows : []);
    } catch {
      if (isCurrent()) setBindings([]);
    }
  }, [projectId, resolvedStore]);

  const saveDefault = useCallback(
    async (
      task: InferenceTask,
      modelId: string,
    ): Promise<ProjectDefaultResult> => {
      if (!projectId) return { ok: false, message: "No project" };
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
        return { ok: true, binding: saved };
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Could not save the project default.";
        setError(message);
        return { ok: false, message };
      } finally {
        setSaving(null);
      }
    },
    [projectId, resolvedStore],
  );

  const clearDefault = useCallback(
    async (task: InferenceTask): Promise<ProjectDefaultResult> => {
      if (!projectId) return { ok: false, message: "No project" };
      setSaving(task);
      setError(null);
      try {
        await clearProjectDefault(resolvedStore, projectId, task);
        // A clear outranks an initial list still in flight; see generationRef.
        generationRef.current += 1;
        setBindings((current) =>
          current.filter((binding) => binding.task !== task),
        );
        return { ok: true, binding: null };
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Could not clear the project default.";
        setError(message);
        return { ok: false, message };
      } finally {
        setSaving(null);
      }
    },
    [projectId, resolvedStore],
  );

  return {
    bindings,
    saving,
    error,
    defaultModelId,
    saveDefault,
    clearDefault,
    refresh,
  };
}
