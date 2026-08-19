/**
 * What a failed read is allowed to take away.
 *
 * Reads default to `retry: false` and `refetchOnWindowFocus: true`, so an
 * offline researcher fails one every time they come back to the tab. A read
 * that never succeeded has nothing to show but its banner; a read that already
 * succeeded has a rendered page behind it, and losing that page to a refresh
 * nobody asked for is the failure these tests exist to keep out.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { queryClient } from "../api/queryClient";
import { clearAccessToken } from "../auth/storage";
import { useServerQuery } from "./useServerQuery";

const KEY = ["test-read"];

/** A read the test switches between answering and failing, as a network does. */
function controllableRead() {
  let succeeds = true;
  let served = 0;
  const read = vi.fn(() => {
    if (!succeeds) return Promise.reject(new Error("network down"));
    served += 1;
    return Promise.resolve(`value ${served}`);
  });
  return {
    read,
    goOffline: () => {
      succeeds = false;
    },
    comeBack: () => {
      succeeds = true;
    },
  };
}

function renderRead(read: () => Promise<string>) {
  const onError = vi.fn(() => "boom");
  const view = renderHook(() =>
    useServerQuery<string>({ key: KEY, tags: ["test"], read, onError }),
  );
  return { view, onError };
}

/** The cache, not the render, is where a background failure lands first. */
async function refetchUntilFailed() {
  await act(async () => {
    await queryClient.refetchQueries({ queryKey: KEY, exact: true });
  });
  await waitFor(() =>
    expect(queryClient.getQueryState(KEY)?.status).toBe("error"),
  );
  await act(async () => {
    await Promise.resolve();
  });
}

describe("useServerQuery", () => {
  beforeEach(() => {
    queryClient.clear();
  });

  it("reports a first read that fails and shows nothing", async () => {
    const { read, goOffline } = controllableRead();
    goOffline();
    const { view, onError } = renderRead(read);

    await waitFor(() => expect(view.result.current.error).toBe("boom"));
    expect(view.result.current.data).toBeNull();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("keeps the loaded value when a background refetch fails", async () => {
    const { read, goOffline } = controllableRead();
    const { view } = renderRead(read);
    await waitFor(() => expect(view.result.current.data).toBe("value 1"));

    goOffline();
    await refetchUntilFailed();

    // The page is still on screen and still true. Blanking it to an error
    // banner is what a refetch on window focus used to do.
    expect(view.result.current.data).toBe("value 1");
    expect(view.result.current.error).toBeNull();
  });

  it("reports a failure over loaded content once, not on every refetch", async () => {
    const { read, goOffline } = controllableRead();
    const { view, onError } = renderRead(read);
    await waitFor(() => expect(view.result.current.data).toBe("value 1"));

    goOffline();
    await refetchUntilFailed();
    await refetchUntilFailed();
    await refetchUntilFailed();

    // Every focus of an offline tab used to be another toast.
    expect(read).toHaveBeenCalledTimes(4);
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("reports again after a read succeeds in between", async () => {
    const { read, goOffline, comeBack } = controllableRead();
    const { view, onError } = renderRead(read);
    await waitFor(() => expect(view.result.current.data).toBe("value 1"));

    goOffline();
    await refetchUntilFailed();
    expect(onError).toHaveBeenCalledTimes(1);

    comeBack();
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: KEY, exact: true });
    });
    await waitFor(() => expect(view.result.current.data).toBe("value 2"));

    goOffline();
    await refetchUntilFailed();

    expect(onError).toHaveBeenCalledTimes(2);
    expect(view.result.current.data).toBe("value 2");
  });

  it("does not hand a stale value to the next key", async () => {
    const { read } = controllableRead();
    const { view } = renderRead(read);
    await waitFor(() => expect(view.result.current.data).toBe("value 1"));

    const failing = renderHook(() =>
      useServerQuery<string>({
        key: ["other-read"],
        tags: ["test"],
        read: () => Promise.reject(new Error("network down")),
        onError: () => "boom",
      }),
    );

    await waitFor(() => expect(failing.result.current.error).toBe("boom"));
    expect(failing.result.current.data).toBeNull();
  });

  it("survives a session boundary landing while the first read is in flight", async () => {
    // The public document page's exact startup: its reads leave before the
    // AuthProvider's session restore settles, and the restore's
    // clearAccessToken() resets the query cache mid-flight. clear() used to
    // destroy the in-flight query and leave the spinner up forever.
    let resolveFirst!: (value: string) => void;
    let served = 0;
    const read = vi.fn(() => {
      served += 1;
      if (served === 1) {
        return new Promise<string>((resolve) => {
          resolveFirst = resolve;
        });
      }
      return Promise.resolve(`value ${served}`);
    });
    const { view } = renderRead(read);
    await waitFor(() => expect(read).toHaveBeenCalledTimes(1));

    act(() => {
      clearAccessToken();
    });
    resolveFirst("value 1");

    await waitFor(() => expect(view.result.current.loading).toBe(false));
    // The first read's value belonged to the session that just ended; what is
    // on screen must come from the refetch that ran after the boundary.
    expect(view.result.current.data).toBe("value 2");
    expect(view.result.current.error).toBeNull();
  });
});
