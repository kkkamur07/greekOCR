"use client";

import { ProtectedRoute } from "../../../components/ProtectedRoute";
import { DevicesPage } from "../../../pages/DevicesPage";

export default function DevicesRoute() {
  return (
    <ProtectedRoute>
      <DevicesPage />
    </ProtectedRoute>
  );
}
