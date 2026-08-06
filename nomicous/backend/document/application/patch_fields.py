"""Which fields each PATCH accepts, and what happens when a caller sends another.

The handlers apply ``model_dump(exclude_unset=True)`` verbatim with ``setattr``, which is
what makes an explicit ``null`` distinguishable from an omitted field. That same
verbatim application is why an unrecognised key has to be refused loudly rather than
dropped: silently ignoring it would let a client believe it had written something.

Over HTTP that refusal is now the request models' own (``extra="forbid"`` on
``DocumentUpdateRequest``, ``BlockPatchRequest`` and ``LinePatchRequest``), which answers
422. Until those were set, pydantic dropped unknown keys before
``model_dump`` ran, so this guard could not fire from any route and the promise above was
not kept by anything.

The guard stays for callers that build the dict themselves - the service methods take
``**updates`` and are reachable from scripts and tests - and as the thing that fails
should a schema ever gain a field this list does not know about.
"""

from __future__ import annotations

from backend.core.exceptions import ValidationError

DOCUMENT_UPDATE_FIELDS = frozenset({"name", "workflow"})
BLOCK_PATCH_FIELDS = frozenset({"order", "box"})
LINE_PATCH_FIELDS = frozenset({"order", "block_id", "baseline", "mask", "points"})


def reject_unknown_fields(
    fields: dict[str, object], allowed: frozenset[str], operation: str
) -> None:
    unknown = set(fields) - allowed
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValidationError(f"Unsupported {operation} field(s): {joined}")
