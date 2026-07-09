import { useNavigate } from 'react-router-dom';
import type { DocumentResponse } from '../../api/client';
import { WorkflowBadge } from '../WorkflowBadge';

type DocumentsTableProps = {
  projectId: string;
  documents: DocumentResponse[];
  loading: boolean;
  emptyText: string;
  onDelete?: (documentId: string) => void;
  deletingDocumentId?: string | null;
};

function formatUpdated(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

export function DocumentsTable({
  projectId,
  documents,
  loading,
  emptyText,
  onDelete,
  deletingDocumentId = null,
}: DocumentsTableProps) {
  const navigate = useNavigate();

  const openDocument = (documentId: string) => {
    void navigate(`/projects/${projectId}/documents/${documentId}`);
  };

  const onRowKeyDown = (event: React.KeyboardEvent, documentId: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openDocument(documentId);
    }
  };

  return (
    <div className="data-panel">
      <table className="data-list" aria-label="Documents in project">
        <thead>
          <tr>
            <th scope="col">Document</th>
            <th scope="col">Status</th>
            <th scope="col">Parts</th>
            <th scope="col">Updated</th>
            <th scope="col">
              <span className="text-muted">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr className="data-list-empty">
              <td colSpan={5}>Loading…</td>
            </tr>
          ) : documents.length === 0 ? (
            <tr className="data-list-empty">
              <td colSpan={5}>{emptyText}</td>
            </tr>
          ) : (
            documents.map((doc) => (
              <tr
                key={doc.id}
                className="data-list-row--clickable"
                tabIndex={0}
                aria-label={`Open document ${doc.name}`}
                onClick={() => openDocument(doc.id)}
                onKeyDown={(event) => onRowKeyDown(event, doc.id)}
              >
                <td>
                  <span className="row-title">{doc.name}</span>
                  <span className="row-sub">{doc.id.slice(0, 8)}</span>
                </td>
                <td className="col-status">
                  <WorkflowBadge workflow={doc.workflow} />
                </td>
                <td className="col-muted">{doc.part_count}</td>
                <td className="col-muted">{formatUpdated(doc.updated_at)}</td>
                <td className="col-action" onClick={(e) => e.stopPropagation()}>
                  <div className="data-list-actions">
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => openDocument(doc.id)}
                    >
                      Open
                    </button>
                    {onDelete && (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm btn--danger-ghost"
                        disabled={deletingDocumentId === doc.id}
                        aria-label={`Delete document ${doc.name}`}
                        onClick={() => onDelete(doc.id)}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
