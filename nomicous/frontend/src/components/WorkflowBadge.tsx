import type { DocumentWorkflow } from '../../api/client';

const WORKFLOW_CLASS: Record<DocumentWorkflow, string> = {
  draft: 'badge-draft',
  published: 'badge-published',
  archived: 'badge-archived',
};

export function WorkflowBadge({ workflow }: { workflow: DocumentWorkflow }) {
  return <span className={`badge ${WORKFLOW_CLASS[workflow]}`}>{workflow}</span>;
}

export function ReviewBadge({ reviewed }: { reviewed: boolean }) {
  return (
    <span className={`badge ${reviewed ? 'badge-reviewed' : 'badge-unreviewed'}`}>
      {reviewed ? 'reviewed' : 'unreviewed'}
    </span>
  );
}
