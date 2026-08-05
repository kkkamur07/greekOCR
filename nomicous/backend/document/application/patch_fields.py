"""Which fields each PATCH accepts, and what happens when a caller sends another.

The handlers apply ``model_dump(exclude_unset=True)`` verbatim with ``setattr``, which is
what makes an explicit ``null`` distinguishable from an omitted field. That same
verbatim application is why an unrecognised key has to be refused loudly rather than
dropped: silently ignoring it would let a client believe it had written something.
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
