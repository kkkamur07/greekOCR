import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import {
  clearProjectDefault,
  saveProjectDefault,
  useProjectModelDefaults,
  type BindingStore,
  type ProjectBinding,
} from "./projectModelDefaults";

function storeWith(bindings: ProjectBinding[]): BindingStore & {
  calls: { create: number; update: number; remove: number };
} {
  const calls = { create: 0, update: 0, remove: 0 };
  let rows = [...bindings];
  return {
    calls,
    list: async () => [...rows],
    create: async (_projectId, task, modelId) => {
      calls.create += 1;
      const created = {
        id: `binding-${task}`,
        task,
        model_id: modelId,
      };
      rows = [...rows.filter((row) => row.task !== task), created];
      return created;
    },
    update: async (_projectId, bindingId, modelId) => {
      calls.update += 1;
      const current = rows.find((row) => row.id === bindingId);
      if (!current) {
        throw new ApiError("Model binding not found", 404);
      }
      const updated = { ...current, model_id: modelId };
      rows = rows.map((row) => (row.id === bindingId ? updated : row));
      return updated;
    },
    remove: async (_projectId, bindingId) => {
      calls.remove += 1;
      if (!rows.some((row) => row.id === bindingId)) {
        throw new ApiError("Model binding not found", 404);
      }
      rows = rows.filter((row) => row.id !== bindingId);
    },
  };
}

describe("saveProjectDefault", () => {
  it("POSTs when the project has no binding for the task", async () => {
    const store = storeWith([]);
    const saved = await saveProjectDefault(
      store,
      "project-1",
      "transcribe",
      "htr-syriac",
    );
    expect(saved).toMatchObject({ task: "transcribe", model_id: "htr-syriac" });
    expect(store.calls).toEqual({ create: 1, update: 0, remove: 0 });
  });

  it("PATCHes the existing binding instead of creating a second row", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-greek" },
    ]);
    const saved = await saveProjectDefault(
      store,
      "project-1",
      "transcribe",
      "htr-syriac",
    );
    expect(saved).toMatchObject({ id: "binding-1", model_id: "htr-syriac" });
    expect(store.calls).toEqual({ create: 0, update: 1, remove: 0 });
  });

  it("leaves a binding that already points at the model alone", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const saved = await saveProjectDefault(
      store,
      "project-1",
      "segment",
      "seg-a",
    );
    expect(saved).toMatchObject({ id: "binding-1" });
    expect(store.calls).toEqual({ create: 0, update: 0, remove: 0 });
  });

  it("refetches and PATCHes when a collaborator wins the create race", async () => {
    const raced = {
      id: "binding-race",
      task: "segment",
      model_id: "seg-b",
    } as const;
    const store = storeWith([{ ...raced }]);
    const create = vi
      .fn()
      .mockRejectedValueOnce(new ApiError("A binding already exists", 409));
    const racing: BindingStore = {
      ...store,
      create,
      list: async () => [{ ...raced }],
    };
    const saved = await saveProjectDefault(
      racing,
      "project-1",
      "segment",
      "seg-a",
    );
    expect(saved).toMatchObject({ id: "binding-race", model_id: "seg-a" });
    expect(store.calls.update).toBe(1);
  });

  it("rethrows a conflict when even the refetch finds no binding", async () => {
    const racing: BindingStore = {
      ...storeWith([]),
      create: async () => {
        throw new ApiError("A binding already exists", 409);
      },
    };
    await expect(
      saveProjectDefault(racing, "project-1", "segment", "seg-a"),
    ).rejects.toMatchObject({ status: 409 });
  });

  it("rethrows failures that are not a create race", async () => {
    const failing: BindingStore = {
      ...storeWith([]),
      create: async () => {
        throw new ApiError("No access", 403);
      },
    };
    await expect(
      saveProjectDefault(failing, "project-1", "segment", "seg-a"),
    ).rejects.toMatchObject({ status: 403 });
  });
});

describe("clearProjectDefault", () => {
  it("deletes the binding for the task", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-greek" },
      { id: "binding-2", task: "segment", model_id: "seg-a" },
    ]);
    await clearProjectDefault(store, "project-1", "transcribe");
    expect(store.calls.remove).toBe(1);
    expect(await store.list("project-1")).toEqual([
      { id: "binding-2", task: "segment", model_id: "seg-a" },
    ]);
  });

  it("does nothing when the task has no binding", async () => {
    const store = storeWith([]);
    await clearProjectDefault(store, "project-1", "segment");
    expect(store.calls.remove).toBe(0);
  });

  it("treats a binding deleted in the meantime as cleared", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const racing: BindingStore = {
      ...store,
      remove: async () => {
        throw new ApiError("Model binding not found", 404);
      },
    };
    await expect(
      clearProjectDefault(racing, "project-1", "segment"),
    ).resolves.toBeUndefined();
  });

  it("rethrows a failure that is not a missing binding", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const failing: BindingStore = {
      ...store,
      remove: async () => {
        throw new ApiError("No access", 403);
      },
    };
    await expect(
      clearProjectDefault(failing, "project-1", "segment"),
    ).rejects.toMatchObject({ status: 403 });
  });
});

describe("useProjectModelDefaults", () => {
  it("loads the bindings and reports each task default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-syriac" },
    ]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac"),
    );
    expect(result.current.defaultModelId("segment")).toBeNull();
  });

  it("saves through the store and moves the default", async () => {
    const store = storeWith([]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await act(async () => {
      await result.current.saveDefault("transcribe", "htr-syriac");
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
    expect(store.calls.create).toBe(1);
  });

  it("reports a failed save and keeps the old default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-greek" },
    ]);
    const failing: BindingStore = {
      ...store,
      update: async () => {
        throw new ApiError("No access", 403);
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", failing),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("transcribe")).toBe("htr-greek"),
    );
    let saved: unknown;
    await act(async () => {
      saved = await result.current.saveDefault("transcribe", "htr-syriac");
    });
    expect(saved).toMatchObject({ ok: false, message: "No access" });
    expect(result.current.error).toMatch(/No access/);
    expect(result.current.defaultModelId("transcribe")).toBe("htr-greek");
  });

  it("clears through the store and drops the default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("segment")).toBe("seg-a"),
    );
    await act(async () => {
      await result.current.clearDefault("segment");
    });
    expect(result.current.defaultModelId("segment")).toBeNull();
    expect(store.calls.remove).toBe(1);
  });

  it("reports a failed clear and keeps the stored default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const failing: BindingStore = {
      ...store,
      remove: async () => {
        throw new ApiError("No access", 403);
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", failing),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("segment")).toBe("seg-a"),
    );
    let cleared: unknown;
    await act(async () => {
      cleared = await result.current.clearDefault("segment");
    });
    expect(cleared).toMatchObject({ ok: false, message: "No access" });
    expect(result.current.defaultModelId("segment")).toBe("seg-a");
  });

  it("keeps a save that lands while the initial list is still in flight", async () => {
    let releaseFirst!: (rows: ProjectBinding[]) => void;
    const firstGate = new Promise<ProjectBinding[]>((resolve) => {
      releaseFirst = resolve;
    });
    let calls = 0;
    const store = storeWith([]);
    const gated: BindingStore = {
      ...store,
      list: async () => {
        calls += 1;
        return calls === 1 ? firstGate : [];
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", gated),
    );
    // The initial list goes first; the save below must be the second call.
    await waitFor(() => expect(calls).toBe(1));
    await act(async () => {
      await result.current.saveDefault("transcribe", "htr-syriac");
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
    await act(async () => {
      releaseFirst([]);
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
  });

  it("refresh re-reads the bindings after an outside write", async () => {
    const store = storeWith([]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await store.create("project-1", "transcribe", "htr-syriac");
    expect(result.current.defaultModelId("transcribe")).toBeNull();
    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
  });

  it("fetches nothing without a project", () => {
    const store = storeWith([]);
    const list = vi.spyOn(store, "list");
    renderHook(() => useProjectModelDefaults(undefined, store));
    expect(list).not.toHaveBeenCalled();
  });
});
