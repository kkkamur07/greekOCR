import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { toast } from '../components/ui/toast';
import { api, type ProjectResponse } from '../api/client';
import { ApiError } from '../api/errors';
import { hasAccessToken, isUnauthorized, navigateToLogin } from '../auth/session';
import { AppPageShell } from '../components/layout/AppPageShell';
import { ProjectsTable } from '../components/projects/ProjectsTable';
import { FormModal } from '../components/ui/FormModal';
import { slugify } from '../utils/slugify';

export function ProjectsPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [userId, setUserId] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newName, setNewName] = useState('');

  const load = useCallback(async () => {
    if (!hasAccessToken()) {
      navigateToLogin(router);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [me, list] = await Promise.all([api.me(), api.listProjects()]);
      setUserId(me.id);
      setUsername(me.username);
      setProjects(list);
    } catch (err) {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return;
      }
      const msg = err instanceof ApiError ? err.message : 'Failed to load projects';
      setProjects([]);
      setUserId(null);
      setUsername(null);
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await api.createProject({ name: newName.trim(), slug: slugify(newName) });
      toast.success('Project created');
      setCreateModalOpen(false);
      setNewName('');
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to create project';
      toast.error(msg);
    } finally {
      setCreating(false);
    }
  };

  const owned = projects.filter((p) => p.owner_id === userId);
  const shared = projects.filter((p) => p.owner_id !== userId);

  const handleDeleteProject = async (projectId: string) => {
    const project = projects.find((item) => item.id === projectId);
    if (!project) return;
    const confirmed = window.confirm(
      `Delete project "${project.name}"? All documents in this project will be removed.`,
    );
    if (!confirmed) return;

    setDeletingProjectId(projectId);
    try {
      await api.deleteProject(projectId);
      toast.success('Project deleted');
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to delete project';
      toast.error(msg);
    } finally {
      setDeletingProjectId(null);
    }
  };

  return (
    <AppPageShell
      currentLabel="Projects"
      username={username}
      title="Projects"
      subtitle="Owned and shared"
      headerActions={
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={() => setCreateModalOpen(true)}
        >
          New project
        </button>
      }
    >
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Projects unavailable</strong>
          {error}
        </div>
      )}

      <p className="section-label" id="owned-label">
        Owned
      </p>
      <ProjectsTable
        id="owned-label"
        caption="Owned projects"
        projects={owned}
        userId={userId}
        loading={loading}
        emptyText="No owned projects yet"
        onDelete={(projectId) => void handleDeleteProject(projectId)}
        deletingProjectId={deletingProjectId}
      />

      <p className="section-label" id="shared-label">
        Shared
      </p>
      <ProjectsTable
        id="shared-label"
        caption="Shared projects"
        projects={shared}
        userId={userId}
        loading={loading}
        emptyText="No shared projects"
        showOwner
      />

      <FormModal
        open={createModalOpen}
        title="New project"
        onClose={() => setCreateModalOpen(false)}
        onSubmit={handleCreate}
        submitLabel="Create"
        loading={creating}
      >
        <div className="field">
          <label htmlFor="project-name">Name</label>
          <input
            id="project-name"
            required
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
        </div>
      </FormModal>
    </AppPageShell>
  );
}
