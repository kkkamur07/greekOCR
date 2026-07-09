import { Navigate, Route, Routes } from 'react-router-dom';
import { ProtectedRoute } from './components/ProtectedRoute';
import { DocumentDetailPage } from './pages/DocumentDetailPage';
import { PageEditorPlaceholderPage } from './pages/PageEditorPlaceholderPage';
import { PublicDocumentPage } from './pages/PublicDocumentPage';
import { LoginPage } from './pages/LoginPage';
import { ProjectDashboardPage } from './pages/ProjectDashboardPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { RegisterPage } from './pages/RegisterPage';
import { hasAccessToken } from './auth/session';

function App() {
  const isAuthed = hasAccessToken();

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route
        path="/projects"
        element={
          <ProtectedRoute>
            <ProjectsPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/projects/:projectId"
        element={
          <ProtectedRoute>
            <ProjectDashboardPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/projects/:projectId/documents/:documentId"
        element={
          <ProtectedRoute>
            <DocumentDetailPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/projects/:projectId/documents/:documentId/parts/:partId"
        element={
          <ProtectedRoute>
            <PageEditorPlaceholderPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/public/projects/:projectId/documents/:documentId"
        element={<PublicDocumentPage />}
      />
      <Route
        path="/"
        element={<Navigate to={isAuthed ? '/projects' : '/login'} replace />}
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
