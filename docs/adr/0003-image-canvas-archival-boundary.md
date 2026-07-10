# ImageCanvas archival boundary

## Status

Recorded

## Context

The legacy `nomicous/frontend/src/components/ImageCanvas/` family has no
current importers. That evidence alone does not make its removal part of every
unrelated cleanup: it may still be useful as historical or research reference
material.

## Decision

This cleanup slice does not delete or relocate `ImageCanvas`. Any archival or
removal must be a separate, explicitly reviewed change that inventories the
component root, `components/`, `hooks/`, styles, and any documentation links
before changing them.

The targeted cleanup may remove independently verified zero-reference files,
such as a duplicate overlay implementation, without expanding to the
`ImageCanvas` family.

## Consequences

- Reviewers can distinguish narrowly proven dead-code removals from a larger
  legacy-component retirement.
- Future archival work has an explicit boundary and inventory requirement.
