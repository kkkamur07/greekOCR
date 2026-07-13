"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthSession } from "../auth/AuthProvider";
import { SessionRestoring } from "../components/SessionRestoring";

export default function HomePage() {
  const router = useRouter();
  const { status } = useAuthSession();

  useEffect(() => {
    if (status !== "restoring") {
      router.replace(status === "authenticated" ? "/projects" : "/login");
    }
  }, [router, status]);

  return (
    <SessionRestoring
      label={
        status === "restoring"
          ? "Restoring your session…"
          : status === "authenticated"
            ? "Opening your projects…"
            : "Redirecting to sign in…"
      }
    />
  );
}
