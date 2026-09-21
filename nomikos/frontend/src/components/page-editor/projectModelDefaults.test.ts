import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import {
  saveProjectDefault,
  useProjectModelDefaults,
  type BindingStore,
  type ProjectBinding,
} from "./projectModelDefaults";

function storeWith(bindings: ProjectBinding[]): BindingStore & {
  calls: { create: number; update: number };
} {
  const calls = { create: 0, update: 0 };
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
    expect(store.calls).toEqual({ create: 1, update: 0 });
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
    expect(store.calls).toEqual({ create: 0, update: 1 });
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
    expect(store.calls).toEqual({ create: 0, update: 0 });
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
    expect(saved).toBeNull();
    expect(result.current.error).toMatch(/No access/);
    expect(result.current.defaultModelId("transcribe")).toBe("htr-greek");
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
