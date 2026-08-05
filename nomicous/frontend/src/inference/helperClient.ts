import { HELPER_BASE_URL } from "./constants";

/** Chromium Local Network Access: mark loopback so HTTPS pages can talk to the helper. */
type LoopbackRequestInit = RequestInit & {
  targetAddressSpace?: "loopback" | "local" | "public";
};

/**
 * Call the local helper on its one known origin.
 *
 * There is deliberately no URL fallback list: a failure here means "no helper",
 * never "try the next port and trust whoever answers".
 */
export function fetchHelper(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const loopbackInit: LoopbackRequestInit = {
    ...init,
    targetAddressSpace: "loopback",
  };
  return fetch(`${HELPER_BASE_URL}${path}`, loopbackInit);
}
