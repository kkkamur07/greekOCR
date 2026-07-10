'use client';

import { ProtectedRoute } from '../../../../../components/ProtectedRoute';
import { DocumentDetailPage } from '../../../../../pages/DocumentDetailPage';

export default function DocumentDetailRoute() {
  return (
    <ProtectedRoute>
      <DocumentDetailPage />
    </ProtectedRoute>
  );
}
