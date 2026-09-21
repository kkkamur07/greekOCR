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
 *
 * `superseded` marks a response the caller must act on in no way at all: a
 * newer write for the same task started while this one was in flight, so its
 * outcome is neither the project's state nor news for the person.
 */
export type ProjectDefaultResult =
  | { ok: true; superseded?: false; binding: ProjectBinding | null }
  | { ok: false; superseded?: false; message: string }
  | { ok: false; superseded: true };

/** How often a clear re-lists after a 404 before it gives up. */
const CLEAR_ATTEMPTS = 3;

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
 *
 * A 404 from the delete means the id went stale between the list and the
 * delete, not that the default is gone: a collaborator may have replaced the
 * binding with a new row for the same task. So re-list and delete what is
 * there now, and report the clear only once a list comes back without one.
 */
export async function clearProjectDefault(
  store: BindingStore,
  projectId: string,
  task: InferenceTask,
): Promise<void> {
  for (let attempt = 0; attempt < CLEAR_ATTEMPTS; attempt += 1) {
    const bindings = await store.list(projectId);
    const existing = bindings.find((binding) => binding.task === task);
    if (!existing) return;
    try {
      await store.remove(projectId, existing.id);
      return;
    } catch (err) {
      if (!(err instanceof ApiError) || err.status !== 404) throw err;
    }
  }
  throw new Error(
    "Could not clear the project default: it keeps being replaced.",
  );
}

/**
 * The project bindings behind the project page's default models card, behind
 * the Workflow menu's pickers, and behind every picker's preselection.
 *
 * `loading` and `loadFailed` are part of the answer, not bookkeeping: an
 * empty `bindings` after a failed read means "we do not know", and a surface
 * that draws it as "No default" tells the researcher something about their
 * project that nobody checked. Callers wait, or say they could not look.
 */
export function useProjectModelDefaults(
  projectId: string | undefined,
  store?: BindingStore,
) {
  const resolvedStore = useMemo(() => store ?? apiBindingStore(), [store]);
  const [bindings, setBindings] = useState<ProjectBinding[]>([]);
  const [loading, setLoading] = useState<boolean>(Boolean(projectId));
  const [loadFailed, setLoadFailed] = useState(false);
  /** The tasks with a write in flight, so one task's save never freezes the other. */
  const [saving, setSaving] = useState<ReadonlySet<InferenceTask>>(
    () => new Set(),
  );
  const [error, setError] = useState<string | null>(null);
  /**
   * Which load is newest. A save that lands while the initial list is still
   * in flight must win over it: without this the late list would overwrite
   * the just-saved default with the server rows it left behind.
   */
  const generationRef = useRef(0);
  /**
   * The newest write per task. A slower older write must not land on top of a
   * newer choice, so a response whose number is no longer the task's latest
   * changes nothing and tells the caller nothing.
   */
  const writeSeqRef = useRef<Partial<Record<InferenceTask, number>>>({});

  const beginWrite = useCallback((task: InferenceTask): number => {
    const seq = (writeSeqRef.current[task] ?? 0) + 1;
    writeSeqRef.current[task] = seq;
    setSaving((current) => new Set(current).add(task));
    setError(null);
    return seq;
  }, []);

  const endWrite = useCallback((task: InferenceTask, seq: number): boolean => {
    const latest = writeSeqRef.current[task] === seq;
    if (latest) {
      setSaving((current) => {
        const next = new Set(current);
        next.delete(task);
        return next;
      });
    }
    return latest;
  }, []);

  const load = useCallback(
    async (id: string, cancelled?: () => boolean): Promise<void> => {
      const generation = ++generationRef.current;
      const isCurrent = () =>
        !cancelled?.() && generationRef.current === generation;
      setLoading(true);
      try {
        // Wrapped so a store that answers with nothing (a test double, an old
        // stub) reads as an empty list rather than crashing the picker.
        const rows = await resolvedStore.list(id);
        if (!isCurrent()) return;
        setBindings(Array.isArray(rows) ? rows : []);
        setLoadFailed(false);
      } catch {
        if (!isCurrent()) return;
        setBindings([]);
        setLoadFailed(true);
      } finally {
        if (isCurrent()) setLoading(false);
      }
    },
    [resolvedStore],
  );

  useEffect(() => {
    if (!projectId) {
      setBindings([]);
      setLoading(false);
      setLoadFailed(false);
      return;
    }
    let cancelled = false;
    void load(projectId, () => cancelled);
    return () => {
      cancelled = true;
    };
  }, [projectId, load]);

  const defaultModelId = useCallback(
    (task: InferenceTask): string | null =>
      bindings.find((binding) => binding.task === task)?.model_id ?? null,
    [bindings],
  );

  /**
   * Re-read the project bindings, for the card's "Try again". A late read
   * never wins over newer state: every read carries the generation guard.
   */
  const refresh = useCallback(async (): Promise<void> => {
    if (!projectId) return;
    await load(projectId);
  }, [projectId, load]);

  const saveDefault = useCallback(
    async (
      task: InferenceTask,
      modelId: string,
    ): Promise<ProjectDefaultResult> => {
      if (!projectId) return { ok: false, message: "No project" };
      const seq = beginWrite(task);
      try {
        const saved = await saveProjectDefault(
          resolvedStore,
          projectId,
          task,
          modelId,
        );
        if (!endWrite(task, seq)) return { ok: false, superseded: true };
        // A save outranks an initial list still in flight; see generationRef.
        generationRef.current += 1;
        setBindings((current) => [
          ...current.filter((binding) => binding.task !== task),
          saved,
        ]);
        setLoadFailed(false);
        return { ok: true, binding: saved };
      } catch (err) {
        if (!endWrite(task, seq)) return { ok: false, superseded: true };
        const message =
          err instanceof Error
            ? err.message
            : "Could not save the project default.";
        setError(message);
        return { ok: false, message };
      }
    },
    [projectId, resolvedStore, beginWrite, endWrite],
  );

  const clearDefault = useCallback(
    async (task: InferenceTask): Promise<ProjectDefaultResult> => {
      if (!projectId) return { ok: false, message: "No project" };
      const seq = beginWrite(task);
      try {
        await clearProjectDefault(resolvedStore, projectId, task);
        if (!endWrite(task, seq)) return { ok: false, superseded: true };
        // A clear outranks an initial list still in flight; see generationRef.
        generationRef.current += 1;
        setBindings((current) =>
          current.filter((binding) => binding.task !== task),
        );
        setLoadFailed(false);
        return { ok: true, binding: null };
      } catch (err) {
        if (!endWrite(task, seq)) return { ok: false, superseded: true };
        const message =
          err instanceof Error
            ? err.message
            : "Could not clear the project default.";
        setError(message);
        return { ok: false, message };
      }
    },
    [projectId, resolvedStore, beginWrite, endWrite],
  );

  return {
    bindings,
    loading,
    loadFailed,
    /** True once a list came back, so a surface may speak about the defaults. */
    known: Boolean(projectId) && !loading && !loadFailed,
    saving,
    error,
    defaultModelId,
    saveDefault,
    clearDefault,
    refresh,
  };
}
