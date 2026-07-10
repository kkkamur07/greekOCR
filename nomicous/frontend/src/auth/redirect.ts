export function authRedirectTarget(
  callbackUrl: string | null | undefined,
): string {
  if (
    !callbackUrl ||
    !callbackUrl.startsWith("/") ||
    callbackUrl.startsWith("//")
  ) {
    return "/projects";
  }

  try {
    const url = new URL(callbackUrl, "http://nomicous.internal");
    return url.origin === "http://nomicous.internal"
      ? `${url.pathname}${url.search}${url.hash}`
      : "/projects";
  } catch {
    return "/projects";
  }
}
