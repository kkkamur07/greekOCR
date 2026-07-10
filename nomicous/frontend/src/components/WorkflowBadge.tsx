import type { DocumentWorkflow } from '../api/client';

export const WORKFLOW_LABEL: Record<DocumentWorkflow, string> = {
  draft: 'Draft',
  published: 'Live',
  archived: 'Archived',
};

const WORKFLOW_CLASS: Record<DocumentWorkflow, string> = {
  draft: 'badge-draft',
  published: 'badge-live',
  archived: 'badge-archived',
};

export function WorkflowBadge({ workflow }: { workflow: DocumentWorkflow }) {
  return (
    <span className={`badge ${WORKFLOW_CLASS[workflow]}`}>{WORKFLOW_LABEL[workflow]}</span>
  );
}

export function ReviewBadge({ reviewed }: { reviewed: boolean }) {
  return (
    <span className={`badge ${reviewed ? 'badge-reviewed' : 'badge-unreviewed'}`}>
      {reviewed ? 'reviewed' : 'unreviewed'}
    </span>
  );
}
