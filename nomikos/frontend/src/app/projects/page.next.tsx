"use client";

import { ProtectedRoute } from "../../components/ProtectedRoute";
import { ProjectsPage } from "../../pages/ProjectsPage";

export default function ProjectsRoute() {
  return (
    <ProtectedRoute>
      <ProjectsPage />
    </ProtectedRoute>
  );
}
