import { Tag } from 'antd';
import type { DocumentWorkflow } from '../api/client';

const WORKFLOW_COLORS: Record<DocumentWorkflow, string> = {
  draft: 'default',
  published: 'green',
  archived: 'orange',
};

export function WorkflowBadge({ workflow }: { workflow: DocumentWorkflow }) {
  return <Tag color={WORKFLOW_COLORS[workflow]}>{workflow}</Tag>;
}
