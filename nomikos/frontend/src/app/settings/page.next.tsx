"use client";

import { ProtectedRoute } from "../../components/ProtectedRoute";
import { SettingsPage } from "../../pages/SettingsPage";

export default function SettingsRoute() {
  return (
    <ProtectedRoute>
      <SettingsPage />
    </ProtectedRoute>
  );
}
