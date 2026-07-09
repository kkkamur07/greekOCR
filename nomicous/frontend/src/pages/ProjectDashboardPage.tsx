import { useCallback, useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { toast } from '../components/ui/toast';
import { api, type DocumentResponse, type ProjectResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { hasAccessToken, isUnauthorized, navigateToLogin } from '../auth/session';
import { AppPageShell } from '../components/layout/AppPageShell';
import { DocumentsTable } from '../components/projects/DocumentsTable';
import { ProjectJobsPanel } from '../components/projects/ProjectJobsPanel';
import { ProjectSettingsPanel } from '../components/sharing/ProjectSettingsPanel';
import { FormModal } from '../components/ui/FormModal';

export function ProjectDashboardPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { projectId } = useParams<{ projectId: string }>();
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [userId, setUserId] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [includeArchived, setIncludeArchived] = useState(false);
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [titlePanelOpen, setTitlePanelOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingProject, setDeletingProject] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newDocName, setNewDocName] = useState('');

  const load = useCallback(async () => {
    if (!projectId) return;
    if (!hasAccessToken()) {
      navigateToLogin(navigate, location);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [me, proj, docs] = await Promise.all([
        api.me(),
        api.getProject(projectId),
        api.listDocuments(projectId, includeArchived),
      ]);
      setUserId(me.id);
      setUsername(me.username);
      setProject(proj);
      setDocuments(docs);
    } catch (err) {
      if (isUnauthorized(err)) {
        navigateToLogin(navigate, location);
        return;
      }
      const msg = err instanceof ApiError ? err.message : 'Failed to load project';
      setProject(null);
      setDocuments([]);
      setError(
        err instanceof ApiError && (err.status === 403 || err.status === 404)
          ? 'This project is not available to your account.'
          : msg,
      );
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }, [projectId, includeArchived, location, navigate]);

  useEffect(() => {
    void load();
  }, [load]);

  const isOwner = Boolean(project && userId && project.owner_id === userId);

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!projectId || !newDocName.trim()) return;
    setCreating(true);
    try {
      const doc = await api.createDocument(projectId, { name: newDocName.trim() });
      toast.success('Document created');
      setCreateModalOpen(false);
      setNewDocName('');
      navigate(`/projects/${projectId}/documents/${doc.id}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to create document';
      toast.error(msg);
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteProject = async () => {
    if (!projectId || !project) return;
    const confirmed = window.confirm(
      `Delete project "${project.name}"? All documents in this project will be removed.`,
    );
    if (!confirmed) return;

    setDeletingProject(true);
    try {
      await api.deleteProject(projectId);
      toast.success('Project deleted');
      navigate('/projects');
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to delete project';
      toast.error(msg);
    } finally {
      setDeletingProject(false);
    }
  };

  const handleDeleteDocument = async (documentId: string) => {
    if (!projectId) return;
    const document = documents.find((item) => item.id === documentId);
    if (!document) return;
    const confirmed = window.confirm(
      `Delete document "${document.name}"? All parts and transcriptions will be removed.`,
    );
    if (!confirmed) return;

    setDeletingDocumentId(documentId);
    try {
      await api.deleteDocument(projectId, documentId);
      toast.success('Document deleted');
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to delete document';
      toast.error(msg);
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const docCountLabel =
    documents.length === 1 ? '1 document' : `${documents.length} documents`;

  return (
    <AppPageShell
      breadcrumb={[
        { label: 'Projects', href: '/projects' },
        { label: project?.name ?? 'Project' },
      ]}
      username={username}
      title={project?.name ?? 'Project'}
      subtitle={project ? docCountLabel : undefined}
      titleEditable={Boolean(isOwner && project && projectId)}
      titlePanelOpen={titlePanelOpen}
      onTitlePanelToggle={() => setTitlePanelOpen((open) => !open)}
      titlePanelLabel="Project settings"
      titlePanel={
        project && projectId ? (
          <ProjectSettingsPanel
            projectId={projectId}
            name={project.name}
            guidelines={project.guidelines ?? null}
            onUpdated={(patch) => {
              setProject((current) =>
                current
                  ? {
                      ...current,
                      name: patch.name,
                      guidelines: patch.guidelines,
                    }
                  : current,
              );
            }}
          />
        ) : null
      }
      headerActions={
        project ? (
          <>
            <label className="field-check">
              <input
                type="checkbox"
                id="show-archived"
                checked={includeArchived}
                onChange={(e) => setIncludeArchived(e.target.checked)}
              />
              Show archived
            </label>
            {isOwner && (
              <button
                type="button"
                className="btn btn-ghost btn-sm btn--danger-ghost"
                disabled={deletingProject}
                onClick={() => void handleDeleteProject()}
              >
                Delete project
              </button>
            )}
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => setCreateModalOpen(true)}
            >
              New document
            </button>
          </>
        ) : undefined
      }
    >
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Project unavailable</strong>
          {error}
        </div>
      )}

      {project && (
        <>
          <DocumentsTable
            projectId={projectId!}
            documents={documents}
            loading={loading}
            emptyText="No documents yet"
            onDelete={(documentId) => void handleDeleteDocument(documentId)}
            deletingDocumentId={deletingDocumentId}
          />

          <ProjectJobsPanel projectId={projectId!} documents={documents} />
        </>
      )}

      <FormModal
        open={createModalOpen}
        title="New document"
        onClose={() => setCreateModalOpen(false)}
        onSubmit={handleCreate}
        submitLabel="Create"
        loading={creating}
      >
        <div className="field">
          <label htmlFor="document-name">Name</label>
          <input
            id="document-name"
            required
            value={newDocName}
            onChange={(e) => setNewDocName(e.target.value)}
          />
        </div>
      </FormModal>
    </AppPageShell>
  );
}
