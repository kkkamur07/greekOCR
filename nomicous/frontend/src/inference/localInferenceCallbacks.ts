export type LocalInferenceCallbacks = {
  onStart: (
    registryModelId: string,
    registryTag?: string,
  ) => void | Promise<void>;
  onEnd: () => void;
  getSignal: () => AbortSignal | undefined;
  shouldFallbackToCloud: () => boolean;
  clearFallbackToCloud: () => void;
};

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
