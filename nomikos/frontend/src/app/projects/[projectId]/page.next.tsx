"use client";

import { ProtectedRoute } from "../../../components/ProtectedRoute";
import { ProjectDashboardPage } from "../../../pages/ProjectDashboardPage";

export default function ProjectDashboardRoute() {
  return (
    <ProtectedRoute>
      <ProjectDashboardPage />
    </ProtectedRoute>
  );
}
