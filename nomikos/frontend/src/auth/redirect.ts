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
    const url = new URL(callbackUrl, "http://nomikos.internal");
    return url.origin === "http://nomikos.internal"
      ? `${url.pathname}${url.search}${url.hash}`
      : "/projects";
  } catch {
    return "/projects";
  }
}
