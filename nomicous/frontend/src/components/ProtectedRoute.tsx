import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthSession } from "../auth/AuthProvider";
import { navigateToLogin } from "../auth/session";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { status } = useAuthSession();
  const authed = status === "authenticated";

  useEffect(() => {
    if (status === "anonymous") {
      navigateToLogin(router);
    }
  }, [router, status]);

  if (status === "restoring" || !authed) {
    return null;
  }

  return <>{children}</>;
}
