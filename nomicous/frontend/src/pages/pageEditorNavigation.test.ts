import { describe, expect, it } from 'vitest';
import { readPageEditorDocument } from './pageEditorNavigation';

const DOCUMENT = {
  id: 'doc-1',
  project_id: 'proj-1',
  name: 'Doc',
  workflow: 'draft' as const,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  part_count: 1,
  parts: [],
};

describe('readPageEditorDocument', () => {
  it('returns the document when route ids match', () => {
    expect(readPageEditorDocument({ document: DOCUMENT }, 'proj-1', 'doc-1')).toBe(DOCUMENT);
  });

  it('returns null when ids do not match', () => {
    expect(readPageEditorDocument({ document: DOCUMENT }, 'proj-2', 'doc-1')).toBeNull();
    expect(readPageEditorDocument({ document: DOCUMENT }, 'proj-1', 'doc-2')).toBeNull();
    expect(readPageEditorDocument(null, 'proj-1', 'doc-1')).toBeNull();
  });
});
