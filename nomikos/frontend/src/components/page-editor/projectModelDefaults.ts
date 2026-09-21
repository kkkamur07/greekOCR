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
 * `superseded` marks a response the caller must act on in no way at all: the
 * hook has moved to another project since the write started, so this outcome
 * is neither the project on screen nor news for the person.
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
 *
 * There is one `saving` for the whole project, not one per task: every
 * control the hook feeds is disabled while a write runs, so a second write
 * cannot start and two writes can never race. `savingTask` is for the label
 * on the row being written, nothing else.
 */
export function useProjectModelDefaults(
  projectId: string | undefined,
  store?: BindingStore,
) {
  const resolvedStore = useMemo(() => store ?? apiBindingStore(), [store]);
  const [bindings, setBindings] = useState<ProjectBinding[]>([]);
  const [loading, setLoading] = useState<boolean>(Boolean(projectId));
  const [loadFailed, setLoadFailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingTask, setSavingTask] = useState<InferenceTask | null>(null);
  const [error, setError] = useState<string | null>(null);
  /**
   * Which load is newest. A write that lands while a list is still in flight
   * must win over it: without this the late list would overwrite the
   * just-saved default with the server rows it left behind. The winner owns
   * `loading` from then on, so an invalidated load never leaves it true.
   */
  const generationRef = useRef(0);
  /**
   * The project the hook is on right now. A write started for one project
   * must not land after the surface has moved to another: its answer is about
   * a project nobody is looking at.
   */
  const projectIdRef = useRef<string | undefined>(projectId);

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
        setLoading(false);
      } catch {
        if (!isCurrent()) return;
        setBindings([]);
        setLoadFailed(true);
        setLoading(false);
      }
    },
    [resolvedStore],
  );

  useEffect(() => {
    projectIdRef.current = projectId;
    // Nothing read from the last project survives the change of project, not
    // its bindings and not a write it had in flight.
    setBindings([]);
    setLoadFailed(false);
    setSaving(false);
    setSavingTask(null);
    setError(null);
    if (!projectId) {
      generationRef.current += 1;
      setLoading(false);
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

  /**
   * Runs one write and settles the hook's state on its answer. A write whose
   * project is no longer the hook's changes nothing at all.
   */
  const runWrite = useCallback(
    async (
      task: InferenceTask,
      write: (id: string) => Promise<ProjectBinding | null>,
      failureMessage: string,
    ): Promise<ProjectDefaultResult> => {
      const writeProjectId = projectId;
      if (!writeProjectId) return { ok: false, message: "No project" };
      setSaving(true);
      setSavingTask(task);
      setError(null);
      const settle = () => {
        if (projectIdRef.current !== writeProjectId) return false;
        setSaving(false);
        setSavingTask(null);
        return true;
      };
      try {
        const saved = await write(writeProjectId);
        if (!settle()) return { ok: false, superseded: true };
        // The write outranks a list still in flight, and owns `loading` from
        // here: see generationRef.
        generationRef.current += 1;
        setBindings((current) => [
          ...current.filter((binding) => binding.task !== task),
          ...(saved ? [saved] : []),
        ]);
        setLoadFailed(false);
        setLoading(false);
        return { ok: true, binding: saved };
      } catch (err) {
        if (!settle()) return { ok: false, superseded: true };
        const message = err instanceof Error ? err.message : failureMessage;
        setError(message);
        return { ok: false, message };
      }
    },
    [projectId],
  );

  const saveDefault = useCallback(
    (task: InferenceTask, modelId: string): Promise<ProjectDefaultResult> =>
      runWrite(
        task,
        (id) => saveProjectDefault(resolvedStore, id, task, modelId),
        "Could not save the project default.",
      ),
    [runWrite, resolvedStore],
  );

  const clearDefault = useCallback(
    (task: InferenceTask): Promise<ProjectDefaultResult> =>
      runWrite(
        task,
        async (id) => {
          await clearProjectDefault(resolvedStore, id, task);
          return null;
        },
        "Could not clear the project default.",
      ),
    [runWrite, resolvedStore],
  );

  return {
    bindings,
    loading,
    loadFailed,
    /** True once a list came back, so a surface may speak about the defaults. */
    known: Boolean(projectId) && !loading && !loadFailed,
    /** One write at a time, so this disables every control the hook feeds. */
    saving,
    /** Which row that write is about, for its own label. */
    savingTask,
    error,
    defaultModelId,
    saveDefault,
    clearDefault,
    refresh,
  };
}
