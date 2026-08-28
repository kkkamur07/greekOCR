import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, type UserResponse } from "../api/client";
import { resourceTags } from "../api/resources";
import { userFacingMessage } from "../api/userFacingError";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { AppPageShell } from "../components/layout/AppPageShell";
import { useServerQuery } from "../hooks/useServerQuery";

/**
 * The account section. There is exactly one entry in it: it exists so
 * `/settings/devices` has a parent to breadcrumb back to, and so the next
 * account-level screen has an obvious home rather than another top-level
 * route.
 */
export function SettingsPage() {
  const router = useRouter();

  const signedIn = hasAccessToken();
  useEffect(() => {
    if (!signedIn) navigateToLogin(router);
  }, [signedIn, router]);

  const { data, error } = useServerQuery<UserResponse>({
    key: signedIn ? ["settings-page"] : null,
    tags: [resourceTags.currentUser],
    read: () => api.me(),
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      return userFacingMessage(err, "Failed to load your account");
    },
  });

  return (
    <AppPageShell
      currentLabel="Settings"
      username={data?.username ?? null}
      title="Settings"
      subtitle={data?.email ?? "Your account"}
    >
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Account details unavailable</strong>
          {error}
        </div>
      )}

      <p className="section-label" id="settings-sections-label">
        Account
      </p>
      <div className="data-panel mb-4">
        <table className="data-list" aria-labelledby="settings-sections-label">
          <tbody>
            <tr>
              <td>
                <Link href="/settings/devices" className="row-title">
                  Devices
                </Link>
                <span className="row-sub">
                  Computers allowed to run jobs on your account, and how to take
                  that back
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </AppPageShell>
  );
}
