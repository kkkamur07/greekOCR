"""The per-line failure policy both architectures have to honour.

Calamari transcribes a page one line crop at a time; BLLA decodes a page into
one refined polygon at a time. Both must isolate a failed line - a single
undecodable crop or a single degenerate contour cannot be allowed to discard
the other thirty-nine lines of the page - and both must stop short of the case
where *nothing* survived.

That second half is the part that is easy to get wrong, and the two paths had
already written it differently. Stated once, here:

    A page that produced at least one line is a partial success, however many
    lines failed. A page that produced none, and had at least one failure, is
    a failed run and re-raises the *first* failure with its original type.

Re-raising rather than returning an empty page is not cosmetic. The original
exception is the only thing that says which kind of failure this was: a broken
artifact raises a ``RuntimeError`` subclass, a bad request raises
``ValueError``, and an **inference agent** puts the reason it carries into the
job callback a researcher reads. Flattening either into "here is a page with no
text" would report a broken model as a successful transcription of a blank
page, and flattening them into one error type would erase the distinction.

Note what does *not* count as a failure: a line the decoder legitimately
declined to emit - BLLA drops candidates with fewer than four ceiling points -
is a skip, not a failure. A page of nothing but skips is genuinely a page with
no text and must return empty rather than raise, which is why the caller
passes a survivor count and a first failure rather than a total.

Both architectures import this module, and both run on ONNX Runtime under
ADR 0006. The policy above does not care which runtime executes it. This
module holds no runtime dependency of its own, which is what lets one rule
govern two differently shaped loops.
"""

from __future__ import annotations


def reraise_if_none_survived(*, survivors: int, first_failure: Exception | None) -> None:
    """Enforce the all-failed half of the per-line isolation policy.

    ``survivors`` is how many units the page actually produced, and
    ``first_failure`` the earliest exception that was isolated (``None`` when
    nothing failed). Raises that exception when a page failed and produced
    nothing; returns quietly otherwise, including when nothing failed and
    nothing was produced.
    """
    if first_failure is not None and survivors == 0:
        raise first_failure


__all__ = ["reraise_if_none_survived"]
