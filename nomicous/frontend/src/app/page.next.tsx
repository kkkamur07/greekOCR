"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthSession } from "../auth/AuthProvider";

export default function HomePage() {
  const router = useRouter();
  const { status } = useAuthSession();

  useEffect(() => {
    if (status !== "restoring") {
      router.replace(status === "authenticated" ? "/projects" : "/login");
    }
  }, [router, status]);

  return null;
}
