"use client";

import { ProtectedRoute } from "../../../../../../../components/ProtectedRoute";
import { PageEditorPlaceholderPage } from "../../../../../../../pages/PageEditorPlaceholderPage";

export default function PageEditorRoute() {
  return (
    <ProtectedRoute>
      <PageEditorPlaceholderPage />
    </ProtectedRoute>
  );
}
