"use client";

import { useEffect } from "react";
import { ProtectedRoute } from "../../components/ProtectedRoute";
import { takePairingTokenFromUrl } from "../../components/devices/pairingToken";
import { PairPage } from "../../pages/PairPage";

export default function PairRoute() {
  // `ProtectedRoute` renders a placeholder while the session is restoring, so
  // on that path `PairPage` never mounts and never gets to empty the fragment
  // itself - and the login redirect that follows would carry the token into a
  // query string. Taking it here covers that case; both calls are idempotent,
  // and a signed-in researcher's `PairPage` effect runs first and wins the
  // race by construction (child effects run before their parent's).
  useEffect(() => {
    takePairingTokenFromUrl();
  }, []);

  return (
    <ProtectedRoute>
      <PairPage />
    </ProtectedRoute>
  );
}
